from src import chicago_crime_predictor as ccp

# Chemin de base du projet dans le conteneur Docker
BASE_DIR = '/opt/airflow'

# Chemins relatifs basés sur le chemin de base dans le conteneur
DATA_DIR = f'{BASE_DIR}/data'
MODELS_DIR = f'{BASE_DIR}/models'

def main():
    # Code pour exécuter les fonctions principales de votre projet
    # Par exemple:
    # - Charger et préparer les données
    # - Entraîner le modèle Prophet
    # - Évaluer le modèle et afficher les résultats
    # Exemple d'appel de fonction depuis votre module src

    # Vous pouvez passer le chemin complet des données et des modèles comme arguments
    obj_predict = ccp.ChicagoCrimePredictor(months_pred=12, data_dir=DATA_DIR)
    obj_predict.df_process()
    obj_predict.update_crime_data()

if __name__ == "__main__":
    main()
