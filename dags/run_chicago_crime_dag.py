from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Fonction wrapper pour votre script principal
def run_main():
    import sys
    sys.path.insert(0, '/opt/airflow/Chicago_Crime')  # Ajoutez le chemin du projet
    from main import main  # Importez la fonction main depuis le fichier main.py
    main()  # Appelez la fonction principale de votre projet



# Arguments par défaut pour le DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),  # Assurez-vous que cette date est dans le passé
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Définir le DAG
dag = DAG(
    'chicago_crime_dag',
    default_args=default_args,
    description='Run Chicago Crime main.py',
    schedule_interval='*/120 * * * *',
    catchup=False,
)

# Définir la tâche qui exécute votre script principal
run_task = PythonOperator(
    task_id='run_main',
    python_callable=run_main,
    dag=dag,
)

# Ordonnancer la tâche
run_task

