# Utilisez une image Python officielle comme image de base
FROM python:3.9

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app/mlflow_server

# Copiez les fichiers nécessaires dans le conteneur

# Installez les dépendances Python si nécessaire
# RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port sur lequel MLflow Server va écouter
EXPOSE 8080

# Commande pour lancer le serveur MLflow lorsque le conteneur démarre
CMD ["mlflow", "server", "--host 0.0.0.0", "--port 8080"]
