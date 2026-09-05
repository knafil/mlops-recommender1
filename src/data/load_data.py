import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_ratings(config: dict) -> pd.DataFrame:
    """Charge et nettoie les ratings MovieLens."""
    logger.info("Chargement des données...")
    
    df = pd.read_csv(
        "data/raw/ml-1m/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python"
    )
    
    logger.info(f"Dataset chargé : {len(df)} ratings")
    logger.info(f"Users : {df['user_id'].nunique()}")
    logger.info(f"Films : {df['movie_id'].nunique()}")
    
    return df

def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Preprocessing de base."""
    # Supprimer les doublons
    df = df.drop_duplicates(subset=["user_id", "movie_id"])
    
    # Normaliser les ratings entre 0 et 1
    df["rating_normalized"] = df["rating"] / 5.0
    
    # Sauvegarder
    output_path = config["data"]["processed_path"]
    df.to_csv(output_path, index=False)
    logger.info(f"Données sauvegardées : {output_path}")
    
    return df
# Exécution
if __name__ == "__main__":
    config = load_config()
    df = load_ratings(config)
    df = preprocess(df, config)

