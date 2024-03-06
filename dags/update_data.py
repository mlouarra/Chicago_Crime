from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import sys


# Définition des arguments par défaut du DAG
default_args = {
    'owner': 'Patricia',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
    'tags': ['MLOps']
}

# Chemin absolu du répertoire parent des DAGs
#parent_dir = os.path.dirname(os.path.abspath(__file__))

# Chemin du répertoire src pour importer chicago_crime_predictor
#sys.path.append(os.path.join(parent_dir, '../src'))

# Définition du DAG
dag = DAG(
    dag_id='update_chicago_crime_data',
    default_args=default_args,
    description='Mise à jour des données Chicago Crime',
    schedule_interval='@weekly'
)

def task_update_crime_data():
    from pathlib import Path

    # Obtenir le chemin complet sans masquage
    chemin_complet = Path().resolve()

    # Imprimer le chemin complet
    print("Chemin new position:", chemin_complet)

    from opt.airflow.src.chicago_crime_predictor import ChicagoCrimePredictor

    ccp = ChicagoCrimePredictor(months_pred=3, data_dir='/opt/airflow/data')
    ccp.update_crime_data()

# Opérateur PythonOperator pour exécuter update_crime_data()
update_crime_data_task = PythonOperator(
    task_id='update_crime_data_task',
    python_callable=task_update_crime_data,
    dag=dag,
)
