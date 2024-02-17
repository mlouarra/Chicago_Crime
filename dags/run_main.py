from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os

# Définition des arguments par défaut du DAG
default_args = {
    'owner': 'Patricia',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 16),  # Date de début du DAG
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,  # Ne pas exécuter les tâches pour les intervalles manqués
    'tags': ['MLOps']  # Tags pour organiser et filtrer les DAGs
}

# Récupérer le chemin absolu du répertoire parent des DAGs
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Définition du DAG
dag = DAG(
    dag_id='run_main',  # Nom du DAG
    default_args=default_args,
    description='Exécute main.py une fois par semaine',
    schedule_interval='@weekly' # Exécution hebdomadaire
)

# Fonction à exécuter par l'opérateur PythonOperator
def run_main_py():
    import subprocess
    script_path = os.path.join(parent_dir, 'main.py')
    subprocess.run(['python3', script_path])

# Opérateur PythonOperator pour exécuter le script main.py
run_main_task = PythonOperator(
    task_id='run_main_script',  # Identifiant de la tâche
    python_callable=run_main_py,  # Fonction à exécuter
    dag=dag,
)

# Définir l'ordre des tâches
run_main_task
