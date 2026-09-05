import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import yaml
import logging
import os
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

def train(config: dict):
    """Entraînement d'un modèle de recommandation basique."""
    
    # Charger les données
    df = pd.read_csv(config["data"]["processed_path"])

    # Extraction de la seed pour la reproductibilité
    seed = config.get("model", {}).get("seed", config.get("data", {}).get("random_seed", 42))
    
    # Split train/test
    train_df, test_df = train_test_split(
        df,
        test_size=config["data"]["test_size"],
        random_state=seed
    )
    
    # Créer une matrice user-item
    # Cette matrice contient une ligne user_id, une colonne movie_id et la valeur du rating
    # Toutes les cases vides seront remplacées par un 0
    train_matrix = train_df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating_normalized",
        fill_value=0
    )
    
    # Modèle basique (baseline) : moyenne par film (sera remplacé par SVD ultérieurement)
    movie_means = train_df.groupby("movie_id")["rating_normalized"].mean()
    
    # Évaluation : on génère les prédictions sur le jeu de test ensuite on calcule la RMSE
    test_df["predicted"] = test_df["movie_id"].map(movie_means).fillna(0.5)
    rmse = np.sqrt(mean_squared_error(
        test_df["rating_normalized"],
        test_df["predicted"]
    ))
    
    logger.info(f"RMSE : {rmse:.4f}")
    # 1. Sauvegarde de l'artefact modèle (.pkl) requis par DVC 
    os.makedirs("models", exist_ok=True) 
    with open("models/recommender.pkl", "wb") as f: 
        pickle.dump(movie_means, f) 
    # 2. Sauvegarde des métriques (.json) requises par DVC 
    metrics = {"rmse": float(rmse)} 
    with open("models/metrics.json", "w") as f: 
         json.dump(metrics, f, indent=2)
    
    return rmse, movie_means
# Enregistrement de l’expérience avec MLflow
if __name__ == "__main__":
    # Premier run MLflow
    mlflow.set_experiment("recommender-baseline")
    
    with mlflow.start_run(run_name="baseline-mean"):
        config = load_config()
        
        if "model" in config:
            # Logger les paramètres
            mlflow.log_params(config["model"])
        
        # Entraîner
        rmse, model = train(config)
        
        # Logger les métriques
        mlflow.log_metric("rmse", rmse)
        
        logger.info("Run MLflow enregistré !")

