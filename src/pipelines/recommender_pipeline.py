from prefect import flow
from prefect.logging import get_run_logger
from src.pipelines.tasks import (
    ingest_and_validate,
    preprocess_and_split,
    build_features,
    train_model,
    evaluate_and_decide
)
@flow(
    name="recommender-ml-pipeline",
    description="Pipeline MLOps complet Prefect pour MovieLens"
)
def recommender_pipeline(
    ratings_path: str = "data/raw/ml-1m/ratings.dat",
    movies_path: str = "data/raw/ml-1m/movies.dat",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    rmse_threshold: float = 1.0,
    mlflow_tracking_uri: str = "http://localhost:5000"
):
    logger = get_run_logger()
    logger.info("Démarrage du Flow Recommender...")
    # 1. Ingestion & Validation
    raw_df, validation_passed = ingest_and_validate(ratings_path, movies_path)
    # Validation Gate
    if not validation_passed:
        logger.error("Validation des données échouée. Arrêt du pipeline.")
        return
    # 2. Preprocessing & Split
    train_df, val_df, test_df = preprocess_and_split(raw_df, val_ratio, test_ratio)
    # 3. Feature Engineering
    train_features = build_features(train_df)
    # 4. Entraînement
    model, metrics = train_model(
        train_features=train_features,
        val_df=val_df,
        mlflow_tracking_uri=mlflow_tracking_uri,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )
    # 5. Évaluation
    should_deploy, test_rmse = evaluate_and_decide(test_df, model, rmse_threshold)
    # 6. Déploiement conditionnel
    if should_deploy:
        logger.info(f"✓ Modèle approuvé pour déploiement (RMSE: {test_rmse:.4f})")
    else:
        logger.warning(f"✗ Modèle rejeté (RMSE: {test_rmse:.4f} >= Seuil: {rmse_threshold})")
if __name__ == "__main__":
    # Exécution directe du flow
    recommender_pipeline()
