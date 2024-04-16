#Dockerfile pour les fichiers du projet
#fichier dockerignore pour exclure airflow, mlflow et les tests, api

FROM python:3.9

WORKDIR /app

COPY . .

 RUN pip install --no-cache-dir -r requirements.txt

# Commande par défaut pour lancer l'application, ou vous pouvez spécifier une commande personnalisée
CMD ['main.py']
