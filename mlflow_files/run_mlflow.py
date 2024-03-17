import mlflow
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from prophet import Prophet
from preprocess import preprocess_data
from evaluate import evaluate_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from optimization import optimize_hyperparameters



# Chemins relatifs basés sur le chemin de base
DATA_DIR = BASE_DIR / 'data'

# Prétraitement des données
df_train, df_test = preprocess_data(DATA_DIR)

# Entraînement du modèle
params = optimize_hyperparameters(df_train, df_test)
#params = {
#    "yearly_seasonality":True,
#    "daily_seasonality":False,
#    "seasonality_prior_scale": 102,
#    "changepoint_prior_scale": 0.05
#}
model = Prophet(**params)
model.fit(df_train)
forecast = model.predict(df_test)
predictions = forecast['yhat']

mae = mean_absolute_error(df_test['y'], predictions)
rmse = mean_squared_error(df_test['y'], predictions, squared=False)
r2 = r2_score(df_test['y'], predictions)

# Évaluation du modèle et suivi dans MLflow
metrics = {"mae": mae, "rmse": rmse, "r2": r2}
artifact_path = "chicago_model_files"
evaluate_model(params, metrics, model, artifact_path)

mlflow.end_run()
if r2 < 0:
    # R² inférieur à 0, démarrer un run MLflow
    with mlflow.start_run(run_name="run_model") as run:
        mlflow.log_params(params)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        mlflow.prophet.log_model(model, artifact_path="model")
        # Log des autres artefacts ou opérations MLflow
else:
    # R² supérieur ou égal à 0, ne pas démarrer de run MLflow
    print("Le R² est supérieur ou égal à 0, aucun run MLflow n'est démarré.")