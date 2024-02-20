import jwt
import passlib
from fastapi import FastAPI, Depends, HTTPException, Header
from src.chicago_crime_predictor import ChicagoCrimePredictor
from src.auth_api import UserLogin, username, verify_password, hashed_password, generate_token, oauth2_scheme, \
    SECRET_KEY, ALGORITHM
from fastapi.security import OAuth2PasswordRequestForm
from pathlib import Path
import joblib

# Importez la classe Logger
from src.logger import Logger

# Configurez le logger
logger = Logger('log.txt').get_logger()

# Instanciez l'application FastAPI
app = FastAPI()

# Chemins relatifs basés sur le chemin de base du fichier main_api.py
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# Chargez le modèle pré-entraîné (supposez que c'est un fichier .pkl)
model_path = MODELS_DIR / 'test_model.pkl'
model = joblib.load(model_path)

# Route d'authentification
@app.post('/login', response_model=dict, tags=["Authentification"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
        Authentifie un utilisateur en vérifiant les informations d'identification et renvoie un token JWT en cas de succès.

        Args:
            form_data (OAuth2PasswordRequestForm): Les informations d'authentification de l'utilisateur.

        Returns:
            dict: Un dictionnaire contenant le token JWT en cas de succès d'authentification.

        Raises:
            HTTPException: En cas d'échec de l'authentification ou d'erreur interne du serveur.
    """
    try:
        if form_data.username == username and verify_password(form_data.password, hashed_password):
            token = generate_token(username)
            logger.info(f"User {form_data.username} logged in successfully.")
            return {'access_token': token, 'token_type': 'bearer'}
        else:
            logger.warning(f"Failed login attempt for user {form_data.username}.")
            raise HTTPException(status_code=401, detail='Échec de l\'authentification')
    except passlib.exc.UnknownHashError:
        logger.error('Internal server error: Unknown hash format.')
        raise HTTPException(status_code=500, detail='Erreur interne du serveur : Format de hachage inconnu')

# Route sécurisée nécessitant un token
@app.get('/secure-data', response_model=dict, tags=["Données sécurisées"])
async def secure_data(current_user: str = Depends(oauth2_scheme)):
    """
        Récupère des données sécurisées pour l'utilisateur actuellement authentifié.

        Args:
            current_user (str): Le token JWT d'autorisation de l'utilisateur.

        Returns:
            dict: Un dictionnaire contenant un message indiquant que les données sont sécurisées pour l'utilisateur actuel.

        Raises:
            HTTPException: En cas d'expiration du token JWT ou de token JWT invalide.
        """
    try:
        payload = jwt.decode(current_user, SECRET_KEY, algorithms=[ALGORITHM])
        current_username = payload['sub']
        logger.info(f"Access to secure data by user {current_username}.")
        return {'message': f'Données sécurisées pour l\'utilisateur {current_username}'}
    except jwt.ExpiredSignatureError:
        logger.warning('Token expired.')
        raise HTTPException(status_code=401, detail='Token expiré')
    except jwt.InvalidTokenError:
        logger.warning('Invalid token.')
        raise HTTPException(status_code=401, detail='Token invalide')

@app.post("/predict/", tags=["Prédiction"])
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
        logger.info(f"Prediction request for incident type {incident_type} in community area {community_area}.")
        # Renvoyer les prédictions
        return {"forecast": forecast.to_dict('records'), "predictions": predictions.to_dict('records')}
    except Exception as e:
        logger.exception("Unexpected error during prediction.")
        # En cas d'erreur, renvoyez un message d'erreur
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/", tags=["Evaluation du modèle"])
async def evaluate():
    """
    Endpoint pour évaluer le modèle et renvoyer les métriques d'évaluation.
    """
    try:
        # Instanciez votre prédicteur (remplacer ... par les paramètres nécessaires)
        predictor = ChicagoCrimePredictor(months_pred=12, data_dir=DATA_DIR)

        # Chargez le modèle si nécessaire
        predictor.model_load(model_path)  # Assurez-vous que la méthode existe dans votre classe

        # Obtenez les données de test et les prédictions
        # Remplacez 'your_test_data' et 'your_predictions' par les méthodes appropriées pour obtenir ces données
        _, test_data = predictor.return_data("ASSAULT", 'Austin')
        _, predictions = predictor.model_predict()

        # Évaluez le modèle en utilisant la méthode model_evaluation
        mae, rmse, r2 = predictor.model_evaluation(test_data, predictions)
        logger.info("Model evaluation completed.")

        # Renvoyez les métriques d'évaluation en réponse
        return {
            "Mean Absolute Error (MAE)": mae,
            "Root Mean Squared Error (RMSE)": rmse,
            "R²": r2
        }
    except Exception as e:
        # En cas d'erreur, renvoyez un message d'erreur
        logger.exception("Unexpected error during model evaluation.")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

