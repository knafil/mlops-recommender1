import json
import logging
import sys
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_data_leakage(df_path: str, target_col: str = "rating", threshold: float = 0.85) -> dict:
    """Détecte les fuites de données (Data Leakage) en excluant la cible et ses dérivées directes."""
    logger.info(f"Analyse du Data Leakage sur : {df_path}")
    df = pd.read_parquet(df_path)
    
    # Exclure la variable cible et ses réécritures directes (ex: note normalisée)
    target_related_cols = [target_col, "rating_normalized"]
    
    numeric_df = df.select_dtypes(include=["float64", "int64", "float32", "int32"])
    if target_col not in numeric_df.columns:
        raise ValueError(f"La colonne cible {target_col} est absente du dataset.")
        
    correlations = numeric_df.corr()[target_col].abs()
    
    suspicious_features = {}
    for col, corr_val in correlations.items():
        if col not in target_related_cols and corr_val > threshold:
            suspicious_features[col] = float(corr_val)
            
    has_leakage = len(suspicious_features) > 0
    
    report = {
        "has_leakage": has_leakage,
        "suspicious_features_count": len(suspicious_features),
        "suspicious_features": suspicious_features,
        "threshold_used": threshold
    }
    
    return report

if __name__ == "__main__":
    report = detect_data_leakage("data/features/clean_features.parquet")
    
    # Sauvegarde du rapport JSON pour DVC
    with open("data/features/leakage_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    if report["has_leakage"]:
        logger.error(f"DATA LEAKAGE DÉTECTÉ ! Features suspectes : {report['suspicious_features']}")
        sys.exit(1)
    else:
        logger.info("✓ Aucun Data Leakage critique n'a été détecté.")
