import mlflow
import os
import schedule
import sys
from pathlib import Path
from optimization import optimize_hyperparameters
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from preprocess import preprocess_data
from evaluate import evaluate_model
from train import train_model
from send_mail import send_email

# Chemins relatifs basés sur le chemin de base
DATA_DIR = BASE_DIR / 'data'

mlflow.create_experiment("Chicago_Model_Evaluation")

# mlflow.set_tracking_uri(uri="http://<host>:<port>")
mlflow.set_tracking_uri(uri="http://mlflow_server:8080")
mlflow.set_experiment("Chicago_Model_Evaluation")

def train_once():
    # Entraîner le modèle une première fois avec paramètres par défaut
    params = {
        "yearly_seasonality": True,
        # "weekly_seasonality": True,
        "daily_seasonality": False,
        "seasonality_prior_scale": 10,  # Augmenter pour une saisonnalité plus flexible
        "changepoint_prior_scale": 0.05  # Diminuer pour des tendances moins flexibles
    }
    df_train, df_test = preprocess_data(DATA_DIR)
    predictions, params, model = train_model(df_train, df_test, params)
    r2, mae, rmse = evaluate_model(predictions, df_test, params, model)
    return r2, mae, rmse, df_train, df_test

# Entraîner le modèle une fois et vérifier si R² est inférieur à 0.5
r2, mae, rmse, df_train, df_test = train_once()

# Vérifier si le R² est inférieur à 0.2 pour commencer les itérations
if r2 < 0.2:

    # Ré-entraînement maximum 5 fois si R² < 0.2
    for i in range(5):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        it = i + 1
        run_name = f"run_train_itération{it}_{current_datetime}"

        params = optimize_hyperparameters(df_train, df_test)
        predictions, params, model = train_model(df_train, df_test, params)
        r2, mae, rmse = evaluate_model(predictions, df_test, params, model)

        artifact_path = "triggered_trained_model"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
            mlflow.prophet.log_model(model, artifact_path)
            # Log des autres artefacts ou opérations MLflow

        if r2 >= 0.2:
            print("Le modèle a un R² satisfaisant.")
            run_id = run.info.run_id
            model_name = "Prophet Model"
            model_version = current_datetime
            # Enregistrer le modèle dans MLflow Registry avec la  version
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, model_name, model_version)

            # Enregistrer les autres artefacts pour affichage streamlit

            mlflow.log_artifact(df_train, "train_data.csv")
            mlflow.log_artifact(df_test, "test_data.csv")
            mlflow.log_artifact(predictions, "predictions.csv")

            break  # Sortir de la boucle si le R² est satisfaisant

    else:  # Cette partie du code sera exécutée si la boucle n'est pas interrompue par un "break"
        # Envoyer un e-mail pour alerter que le R² est toujours insatisfaisant après les itérations
        sender_email = "mlopsaug23@gmail.com"
        sender_password = os.environ.get("EMAIL_PASSWORD")
        receiver_email = "mlopsaug23@gmail.com"
        subject = "Alerte : Score R² inférieur à 0.2 après 5 itérations"
        body = "Le score R² est toujours inférieur à 0.2 après 5 itérations. Veuillez vérifier le modèle."
        send_email(sender_email, sender_password, receiver_email, subject, body)

else:
    print("Le modèle initial a un R² satisfaisant.")
mlflow.end_run()






