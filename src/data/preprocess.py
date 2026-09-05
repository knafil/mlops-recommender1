import pandas as pd
import numpy as np
import yaml
import logging
import sys
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
def load_and_preprocess(config):
    """Charge et préprocesse les données MovieLens."""
    logger.info("Chargement des ratings...")
    ratings = pd.read_csv(
        "data/raw/ml-1m/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python"
    )
    logger.info("Chargement des films...")
    movies = pd.read_csv(
        "data/raw/ml-1m/movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1"
    )
    # Preprocessing
    logger.info("Preprocessing...")
    
    # Supprimer les doublons
    ratings = ratings.drop_duplicates(subset=["user_id", "movie_id"])
    # Normaliser les ratings
    ratings["rating_normalized"] = ratings["rating"] / 5.0
    # Ajouter les genres
    df = ratings.merge(movies[["movie_id", "genres"]], on="movie_id", how="left")
    # Split temporal train/test
    threshold = df["timestamp"].quantile(
        1 - config["data"]["test_size"]
    )
    train = df[df["timestamp"] <= threshold]
    test = df[df["timestamp"] > threshold]
    # Stats
    logger.info(f"Train : {len(train)} ratings")
    logger.info(f"Test  : {len(test)} ratings")
    logger.info(f"Users : {df['user_id'].nunique()}")
    logger.info(f"Films : {df['movie_id'].nunique()}")
    logger.info(f"Sparsité : {1 - len(df) / (df['user_id'].nunique() * df['movie_id'].nunique()):.2%}")
    # Sauvegarder
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/ratings_clean.csv", index=False)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    # Métriques de qualité
    metrics = {
        "n_ratings": len(df),
        "n_users": int(df["user_id"].nunique()),
        "n_movies": int(df["movie_id"].nunique()),
        "n_train": len(train),
        "n_test": len(test),
        "sparsity": float(1 - len(df) / (df["user_id"].nunique() * df["movie_id"].nunique()))
    }
    # Sauvegarder les métriques
    import json
    with open("data/processed/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Preprocessing terminé !")
    return metrics
if __name__ == "__main__":
    config = load_config()
    metrics = load_and_preprocess(config)
    print(metrics)

