import mlflow
#from mlflow import MlflowClient

def evaluate_model(params, metrics, model, artifact_path, run_name="run_model"):
   # client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
    chicago_experiment = mlflow.set_experiment("Chicago_Models")

    #with mlflow.start_run(run_name=run_name) as run:
        #mlflow.log_params(params)
        #mlflow.log_metrics(metrics)
        #mlflow.prophet.log_model(model, artifact_path=artifact_path)
        #mlflow.log_artifact(tmp_file_path)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.prophet.log_model(model, artifact_path=artifact_path)