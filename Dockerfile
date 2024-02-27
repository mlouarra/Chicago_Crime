FROM apache/airflow:2.8.1-python3.9

USER root

# Installer les dépendances nécessaires pour prophet
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        python3-dev \
        libpq-dev \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt .

#WORKDIR /opt/airflow/

RUN pip install --no-cache-dir -r requirements.txt

# Copier des dossiers dans le conteneur Airflow
#COPY ./dags /opt/airflow/dags
#COPY ./src /opt/airflow/src
#COPY main.py /opt/airflow/src
#COPY ./plugins /opt/airflow/plugins
#COPY ./config /opt/airflow/config

#ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
#ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/src"

