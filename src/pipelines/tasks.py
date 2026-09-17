from prefect import task
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, Tuple
# ══════════════════════════════════════════════
# TASK 1 — Ingestion et validation des données
# ══════════════════════════════════════════════
@task(name="Ingest & Validate", retries=2, retry_delay_seconds=30)
def ingest_and_validate(ratings_path: str, movies_path: str) -> Tuple[pd.DataFrame, bool]:
    """Charge les ratings MovieLens et valide leur qualité."""
    print("Chargement des données...")
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python"
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1"
    )
    validation_results = {
        "n_ratings": len(ratings),
        "n_users": int(ratings["user_id"].nunique()),
        "n_movies": int(ratings["movie_id"].nunique()),
        "null_ratings": int(ratings["rating"].isnull().sum()),
        "rating_range_valid": bool(ratings["rating"].between(1, 5).all()),
        "no_duplicates": bool(not ratings.duplicated(subset=["user_id", "movie_id"]).any())
    }
    validation_passed = (
        validation_results["null_ratings"] == 0 and
        validation_results["rating_range_valid"] and
        validation_results["no_duplicates"] and
        validation_results["n_ratings"] > 1000
    )
    df = ratings.merge(movies[["movie_id", "genres"]], on="movie_id", how="left")
    print(f"Validation : {'✓ PASSÉE' if validation_passed else '✗ ÉCHOUÉE'}")
    print(f"Ratings    : {validation_results['n_ratings']:,}")
    return df, validation_passed
# ══════════════════════════════════════════════
# TASK 2 — Preprocessing et split
# ══════════════════════════════════════════════
@task(name="Preprocess & Split")
def preprocess_and_split(
    df: pd.DataFrame, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nettoyage et split temporel."""
    df = df.copy()
    df["rating_normalized"] = df["rating"] / 5.0
    df = df.drop_duplicates(subset=["user_id", "movie_id"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test
# ══════════════════════════════════════════════
# TASK 3 — Feature Engineering
# ══════════════════════════════════════════════
@task(name="Build Features", retries=1, retry_delay_seconds=60)
def build_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Construction des features sans leakage temporel."""
    df = train_df.sort_values("timestamp").copy()
    # Features utilisateur
    df["user_avg_rating"] = df.groupby("user_id")["rating"].transform(lambda x: x.expanding().mean().shift(1))
    df["user_n_ratings"] = df.groupby("user_id")["rating"].transform(lambda x: x.expanding().count().shift(1))
    df["user_std_rating"] = df.groupby("user_id")["rating"].transform(lambda x: x.expanding().std().shift(1))
    # Features film
    df["movie_avg_rating"] = df.groupby("movie_id")["rating"].transform(lambda x: x.expanding().mean().shift(1))
    df["movie_n_ratings"] = df.groupby("movie_id")["rating"].transform(lambda x: x.expanding().count().shift(1))
    # Derivations
    df["rating_vs_user_avg"] = df["rating"] - df["user_avg_rating"]
    df["rating_vs_movie_avg"] = df["rating"] - df["movie_avg_rating"]
    # Recency
    ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
    df["recency_score"] = (df["timestamp"] - ts_min) / (ts_max - ts_min) if ts_max != ts_min else 0
    
    # Imputation
    for col in ["user_avg_rating", "movie_avg_rating"]:
        df[col] = df[col].fillna(df["rating"].mean())
    for col in ["user_n_ratings", "movie_n_ratings", "user_std_rating", "rating_vs_user_avg", "rating_vs_movie_avg"]:
        df[col] = df[col].fillna(0)
    return df
# ══════════════════════════════════════════════
# TASK 4 — Entraînement du modèle
# ══════════════════════════════════════════════
@task(name="Train Model", retries=2, retry_delay_seconds=60, timeout_seconds=3600)
def train_model(
    train_features: pd.DataFrame,
    val_df: pd.DataFrame,
    mlflow_tracking_uri: str = "http://localhost:5000",
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1
) -> Tuple[xgb.XGBRegressor, Dict[str, float]]:
    """Entraînement du modèle XGBoost avec suivi MLflow."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("prefect-recommender")
    feature_cols = [
        "user_avg_rating", "user_n_ratings", "user_std_rating",
        "movie_avg_rating", "movie_n_ratings",
        "rating_vs_user_avg", "rating_vs_movie_avg", "recency_score"
    ]
    available = [c for c in feature_cols if c in train_features.columns]
    X_train, y_train = train_features[available].fillna(0), train_features["rating"]
    # Imputation minimale pour la validation
    X_val = val_df.copy()
    for col in available:
        if col not in X_val.columns:
            X_val[col] = 0
    X_val = X_val[available].fillna(0)
    y_val = val_df["rating"]
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "random_state": 42
    }
    try:
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            model = xgb.XGBRegressor(**params, verbosity=0)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)      
            y_pred = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            mae = float(mean_absolute_error(y_val, y_pred))
            mlflow.log_metrics({"rmse": rmse, "mae": mae})
            mlflow.sklearn.log_model(model, "model")
    except Exception as e:
        print(f"Avertissement MLflow indisponible ({e}), poursuite locale...")
        model = xgb.XGBRegressor(**params, verbosity=0)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
    print(f"Model Trained | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return model, {"rmse": rmse, "mae": mae}
# ══════════════════════════════════════════════
# TASK 5 — Évaluation et décision de déploiement
# ══════════════════════════════════════════════
@task(name="Evaluate & Decide")
def evaluate_and_decide(
    test_df: pd.DataFrame,
    model: xgb.XGBRegressor,
    rmse_threshold: float = 1.0
) -> Tuple[bool, float]:
    """Évaluation sur le test set et validation du déploiement."""
    feature_cols = [
        "user_avg_rating", "user_n_ratings", "user_std_rating",
        "movie_avg_rating", "movie_n_ratings",
        "rating_vs_user_avg", "rating_vs_movie_avg", "recency_score"
    ]
    X_test = test_df.copy()
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0      
    X_test = X_test[feature_cols].fillna(0)
    y_test = test_df["rating"]
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    deploy = rmse < rmse_threshold
    print(f"Test RMSE: {rmse:.4f} | Seuil: {rmse_threshold} | Déploiement: {deploy}")
    return deploy, rmse
