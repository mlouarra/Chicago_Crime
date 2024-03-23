import mlflow
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# from mlflow import MlflowClient

artifact_path = "evaluated_model"
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def evaluate_model(predictions, df_test, params, model):
    # client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
    # Mettre nom du conteneur mlflow ui

    # Évaluation du modèle et suivi dans MLflow
    # metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    run_name = f"run_evaluation_{current_datetime}"
    mae = mean_absolute_error(df_test['y'], predictions)
    rmse = mean_squared_error(df_test['y'], predictions, squared=False)
    r2 = r2_score(df_test['y'], predictions)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        #mlflow.log_artifact("/artifacts", artifact_path)
        mlflow.prophet.log_model(model, artifact_path)
        mlflow.end_run()

    return r2, mae, rmse
