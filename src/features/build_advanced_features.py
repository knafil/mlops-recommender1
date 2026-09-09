import os
import json
import logging
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_embeddings(df: pd.DataFrame, n_components: int = 10) -> tuple:
    """Génère des embeddings utilisateurs et films via SVD (Matrix Factorization)."""
    logger.info("Génération des embeddings SVD...")
    
    # Création de la matrice d'interaction User-Movie
    user_u = df["user_id"].unique()
    movie_u = df["movie_id"].unique()
    
    user_map = {id_: i for i, id_ in enumerate(user_u)}
    movie_map = {id_: i for i, id_ in enumerate(movie_u)}
    
    rows = df["user_id"].map(user_map)
    cols = df["movie_id"].map(movie_map)
    
    ratings_sparse = csr_matrix((df["rating"], (rows, cols)), shape=(len(user_u), len(movie_u)))
    
    # Décomposition SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_emb_matrix = svd.fit_transform(ratings_sparse)
    movie_emb_matrix = svd.components_.T
    
    # DataFrames des Embeddings
    user_emb_df = pd.DataFrame(user_emb_matrix, columns=[f"user_emb_{i}" for i in range(n_components)])
    user_emb_df["user_id"] = user_u
    
    movie_emb_df = pd.DataFrame(movie_emb_matrix, columns=[f"movie_emb_{i}" for i in range(n_components)])
    movie_emb_df["movie_id"] = movie_u
    
    # Ajout du timestamp requis par Feast
    current_time = pd.Timestamp.now(tz="UTC")
    user_emb_df["event_timestamp"] = current_time
    movie_emb_df["event_timestamp"] = current_time
    
    return user_emb_df, movie_emb_df

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule des features glissantes par utilisateur."""
    logger.info("Calcul des features temporelles et glissantes...")
    df = df.sort_values(["user_id", "timestamp"]).copy()
    
    # Convertir en datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    
    # Compter les interactions glissantes (agrégation)
    df["user_ratings_last_30d"] = df.groupby("user_id")["rating"].transform(lambda x: x.rolling(30, min_periods=1).count())
    
    return df

if __name__ == "__main__":
    logger.info("Chargement du dataset d'entraînement...")
    train = pd.read_csv("data/processed/train.csv")
    
    # 1. Génération des Embeddings (4 composantes pour matcher le schéma Feast)
    user_emb, movie_emb = generate_embeddings(train, n_components=4)
    
    # 2. Features glissantes
    train_rolling = compute_rolling_features(train)
    
    # 3. Sauvegarde au format Parquet pour Feast
    os.makedirs("data/features", exist_ok=True)
    user_emb.to_parquet("data/features/user_embeddings.parquet", index=False)
    movie_emb.to_parquet("data/features/movie_embeddings.parquet", index=False)
    train_rolling.to_parquet("data/features/clean_features.parquet", index=False)
    
    logger.info("Features avancées générées avec succès !")
