from src import chicago_crime_predictor as ccp

from pathlib import Path
from src import chicago_crime_predictor as ccp

# Chemin de base du projet
BASE_DIR = Path(__file__).resolve().parent

# Chemins relatifs basés sur le chemin de base
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'


def main():
    # Code pour exécuter les fonctions principales de votre projet
    # Par exemple:
    # - Charger et préparer les données
    # - Entraîner le modèle Prophet
    # - Évaluer le modèle et afficher les résultats
    # Exemple d'appel de fonction depuis votre module src

    obj_predict = ccp.ChicagoCrimePredictor(months_pred=12, data_dir=DATA_DIR)
    obj_predict.df_process()
    obj_predict.update_crime_data()
    df_train, df_test = obj_predict.return_data("ASSAULT", 'Austin')

    obj_predict.model_train(df_train)
    forecast, predictions = obj_predict.model_predict()
    obj_predict.model_save(MODELS_DIR/'test_model.pkl')
    obj_predict.model_evaluation(df_test, predictions)
    obj_predict.model_visualization(df_train, df_test, predictions)
if __name__ == "__main__":
    main()