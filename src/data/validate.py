import pandas as pd
import great_expectations as gx
from great_expectations.expectations.expectation import Expectation
import json
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_ratings(data_path: str = "data/processed/train.csv") -> dict:
    """Valide la qualité du dataset MovieLens via l'API Fluent GX 1.x."""
    logger.info(f"Chargement des données : {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Fichier introuvable : {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    # 1. Contexte éphémère (GX 1.x+)
    context = gx.get_context(mode="ephemeral")
    
    # 2. Configuration du Data Asset
    data_source = context.data_sources.add_pandas("pandas_datasource")
    data_asset = data_source.add_dataframe_asset(name="ratings_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_def")
    
    # 3. Création de la Suite d'Expectations
    suite = context.suites.add(gx.ExpectationSuite(name="ratings_suite"))
    
    # Liste des règles de qualité
    expectations = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="movie_id"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="rating"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="timestamp"),
        gx.expectations.ExpectColumnValuesToBeBetween(column="rating", min_value=1, max_value=5),
        gx.expectations.ExpectColumnValuesToBeBetween(column="rating_normalized", min_value=0.0, max_value=1.0),
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1000, max_value=2000000)
    ]
    
    for exp in expectations:
        suite.add_expectation(exp)
        
    # 4. Exécution de la validation
    validation_def = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="ratings_validation",
            data=batch_definition,
            suite=suite
        )
    )
    
    validation_result = validation_def.run(batch_parameters={"dataframe": df})
    
    # 5. Parsing des résultats
    results = validation_result.results
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    success_rate = passed / len(results) if len(results) > 0 else 0
    
    details = []
    for r in results:
        col = r.expectation_config.kwargs.get("column", "table")
        details.append({
            "expectation": r.expectation_config.type,
            "success": r.success,
            "column": col
        })
        if not r.success:
            logger.warning(f"❌ Échec : {r.expectation_config.type} sur {col}")

    summary = {
        "total_expectations": len(results),
        "passed": passed,
        "failed": failed,
        "success_rate": success_rate,
        "details": details
    }
    
    logger.info(f"Résultat validation : {passed}/{len(results)} règles validées ({success_rate:.0%})")
    return summary

if __name__ == "__main__":
    data_file = "data/processed/train.csv"
    result = validate_ratings(data_file)
    
    os.makedirs("data/processed", exist_ok=True)
    report_path = "data/processed/validation_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Rapport sauvegardé dans {report_path}")
    
    if result["success_rate"] < 0.8:
        logger.error("Qualité des données insuffisante : pipeline arrêté !")
        sys.exit(1)
    
    logger.info("✅ Quality Gate validé avec succès !")
