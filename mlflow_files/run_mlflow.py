import mlflow
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)
sys.path.append(str(BASE_DIR))

from prophet import Prophet
from src import chicago_crime_predictor as ccp
from preprocess import preprocess_data
from visualize import visualize_results
from evaluate import evaluate_model


# Chemins relatifs basés sur le chemin de base
DATA_DIR = BASE_DIR / 'data'

# Prétraitement des données
df_train, df_test = preprocess_data(DATA_DIR)

# Entraînement du modèle
params = {
    "yearly_seasonality":True,
    "daily_seasonality":False,
    "seasonality_prior_scale": 102,
    "changepoint_prior_scale": 0.05
}
model = Prophet(**params)
model.fit(df_train)
obj_predict = ccp.ChicagoCrimePredictor(months_pred=12, data_dir=DATA_DIR)
forecast, predictions = obj_predict.model_predict()
mae, rmse, r2 = obj_predict.model_evaluation(df_test, predictions)

# Visualisation des résultats
tmp_file_path = visualize_results(df_train, df_test, predictions)

# Évaluation du modèle et suivi dans MLflow
metrics = {"mae": mae, "rmse": rmse, "r2": r2}
artifact_path = "chicago_model_files"
evaluate_model(params, metrics, model, artifact_path)
