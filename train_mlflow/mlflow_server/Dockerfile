# Utilisez une image Python officielle comme image de base
FROM python:3.9

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app/mlflow_server

# Copiez les fichiers nécessaires dans le conteneur
COPY mlflow_server .

# Installez les dépendances Python si nécessaire
# RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port sur lequel MLflow Server va écouter
EXPOSE 5000

# Commande pour lancer le serveur MLflow lorsque le conteneur démarre
CMD ["python", "run_train.py"]
