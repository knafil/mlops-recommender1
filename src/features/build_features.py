import pandas as pd
import numpy as np
import os
import json
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features au niveau utilisateur."""
    logger.info("Construction des features utilisateur...")
    
    user_features = df.groupby("user_id").agg(
        n_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        min_rating=("rating", "min"),
        max_rating=("rating", "max"),
        first_rating=("timestamp", "min"),
        last_rating=("timestamp", "max"),
        n_unique_movies=("movie_id", "nunique"),
    ).reset_index()
    
    # Features dérivées
    user_features["rating_range"] = user_features["max_rating"] - user_features["min_rating"]
    user_features["activity_span_days"] = (
        (user_features["last_rating"] - user_features["first_rating"]) / (60 * 60 * 24)
    )
    
    # Horodatage requis pour l'ingestion Feast (Étape 3)
    user_features["datetime"] = pd.to_datetime(user_features["last_rating"], unit="s", utc=True)
    
    # Remplir les NaN
    user_features["std_rating"] = user_features["std_rating"].fillna(0)
    user_features = user_features.fillna(0)
    
    logger.info(f"Features utilisateur : {len(user_features)} users, {len(user_features.columns)} features")
    return user_features

def build_movie_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features au niveau film."""
    logger.info("Construction des features film...")
    
    movie_features = df.groupby("movie_id").agg(
        n_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        pct_high_rating=("rating", lambda x: (x >= 4).mean()),
        pct_low_rating=("rating", lambda x: (x <= 2).mean()),
    ).reset_index()
    
    # Score de popularité normalisé (sécurisé contre la division par zéro)
    min_r = movie_features["n_ratings"].min()
    max_r = movie_features["n_ratings"].max()
    denom = max_r - min_r
    
    if denom > 0:
        movie_features["popularity_score"] = (movie_features["n_ratings"] - min_r) / denom
    else:
        movie_features["popularity_score"] = 0.0
    
    # Horodatage requis pour l'ingestion Feast (Étape 3)
    movie_features["created_timestamp"] = pd.Timestamp.now(tz="UTC")
    
    # Remplir les NaN
    movie_features = movie_features.fillna(0)
    
    logger.info(f"Features film : {len(movie_features)} films, {len(movie_features.columns)} features")
    return movie_features

def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features d'interaction utilisateur-film."""
    logger.info("Construction des features d'interaction...")
    
    user_avg = df.groupby("user_id")["rating"].transform("mean")
    df["rating_centered"] = df["rating"] - user_avg
    
    movie_avg = df.groupby("movie_id")["rating"].transform("mean")
    df["rating_vs_movie_avg"] = df["rating"] - movie_avg
    
    return df[["user_id", "movie_id", "rating", "rating_centered", "rating_vs_movie_avg", "timestamp"]]

if __name__ == "__main__":
    config = load_config()
    
    logger.info("Chargement des données d'entraînement...")
    train = pd.read_csv("data/processed/train.csv")
    
    user_features = build_user_features(train)
    movie_features = build_movie_features(train)
    interaction_features = build_interaction_features(train)
    
    os.makedirs("data/features", exist_ok=True)
    user_features.to_parquet("data/features/user_features.parquet", index=False)
    movie_features.to_parquet("data/features/movie_features.parquet", index=False)
    interaction_features.to_parquet("data/features/interaction_features.parquet", index=False)
    
    metrics = {
        "n_user_features": len(user_features.columns) - 1,
        "n_movie_features": len(movie_features.columns) - 1,
        "n_interactions": len(interaction_features)
    }
    
    with open("data/features/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Feature engineering terminé avec succès !")
    print(json.dumps(metrics, indent=2))
