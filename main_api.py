from fastapi import FastAPI, HTTPException
from src.chicago_crime_predictor import ChicagoCrimePredictor
from pathlib import Path
import joblib

# Instanciez l'application FastAPI
app = FastAPI()

# Chemins relatifs basés sur le chemin de base du fichier main_api.py
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
MODELS_DIR = BASE_DIR / 'models'

# Chargez le modèle pré-entraîné (supposez que c'est un fichier .pkl)
model_path = MODELS_DIR / 'test_model.pkl'
model = joblib.load(model_path)


@app.post("/predict/")
async def predict(incident_type: str, community_area: str):
    """
    Endpoint pour faire des prédictions basées sur le type d'incident et la zone communautaire.
    """
    try:
        # Créez une instance de votre prédicteur avec le modèle chargé
        predictor = ChicagoCrimePredictor(months_pred=12, data_dir=DATA_DIR)
        predictor.model = model

        # Appel des méthodes de prédiction
        df_train, df_test = predictor.return_data(incident_type, community_area)
        forecast, predictions = predictor.model_predict()

        # Renvoyer les prédictions
        return {"forecast": forecast.to_dict('records'), "predictions": predictions.to_dict('records')}
    except Exception as e:
        # En cas d'erreur, renvoyez un message d'erreur
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/")
async def evaluate():
    """
    Endpoint pour évaluer le modèle et renvoyer les métriques d'évaluation.
    """
    try:
        # Instanciez votre prédicteur (remplacer ... par les paramètres nécessaires)
        predictor = ChicagoCrimePredictor(...)

        # Chargez le modèle si nécessaire
        predictor.load_model(model_path)  # Assurez-vous que la méthode existe dans votre classe

        # Obtenez les données de test et les prédictions
        # Remplacez 'your_test_data' et 'your_predictions' par les méthodes appropriées pour obtenir ces données
        test_data = predictor.get_test_data()  # À implémenter
        predictions = predictor.get_predictions(test_data)  # À implémenter

        # Évaluez le modèle en utilisant la méthode model_evaluation
        mae, rmse, r2 = predictor.model_evaluation(test_data, predictions)

        # Renvoyez les métriques d'évaluation en réponse
        return {
            "Mean Absolute Error (MAE)": mae,
            "Root Mean Squared Error (RMSE)": rmse,
            "R²": r2
        }
    except Exception as e:
        # En cas d'erreur, renvoyez un message d'erreur
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

