# Imports librairies
import pandas as pd
import mlflow

import mlflow.prophet
from prophet import Prophet
import plotly.io as pio
from pathlib import Path
from mlflow import MlflowClient


BASE_DIR = Path(__file__).resolve().parent

from src import chicago_crime_predictor as ccp

# Chemins relatifs basés sur le chemin de base
DATA_DIR = BASE_DIR / 'data'
#MODELS_DIR = BASE_DIR / 'models'

# Define tracking_uri
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# Define experiment name, run name and artifact_path name
chicago_experiment = mlflow.set_experiment("Chicago_Models")
run_name = "run_model"
artifact_path = "chicago_model_files"

# Import Database
obj_predict = ccp.ChicagoCrimePredictor(months_pred=12, data_dir=DATA_DIR)
obj_predict.df_process()
df_train, df_test = obj_predict.return_data("ASSAULT", 'Austin')

# Train model
params = {
    "yearly_seasonality":True,
    "daily_seasonality":False,
    "seasonality_prior_scale": 102,
    "changepoint_prior_scale": 0.05
}
model = Prophet(**params)
model.fit(df_train)
forecast, predictions = obj_predict.model_predict()
mae, rmse, r2 = obj_predict.model_evaluation(df_test, predictions)

# Obtenez la figure à partir de la fonction model_visualization()
fig = obj_predict.model_visualization(df_train, df_test, predictions)

# Enregistrez la figure dans un fichier temporaire
tmp_file_path = "model_visualization.html"
pio.write_html(fig, tmp_file_path)

metrics = {"mae": mae, "rmse": rmse, "r2": r2}
# Store information in tracking server
with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.prophet.log_model(
        sk_model=model, artifact_path=artifact_path
    )
    # Log artifacts
    mlflow.log_artifact(tmp_file_path)
