# Utiliser une image d'Airflow comme base
FROM apache/airflow:2.8.1-python3.9

# Définir l'argument de l'utilisateur (changez selon votre cas)
ARG AIRFLOW_USER_HOME=/opt/airflow

# Définir le répertoire de travail
WORKDIR ${AIRFLOW_USER_HOME}

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Mettre à jour pip et installer les dépendances Python spécifiées dans requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
