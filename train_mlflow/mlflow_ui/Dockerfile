FROM python:3.9

# Copier le code source et le répertoire mlruns dans le conteneur
COPY . /app
VOLUME /app/mlruns

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances
RUN pip install mlflow

# Exposer le port pour MLflow UI
EXPOSE 5000

# Commande pour lancer MLflow UI
CMD ["mlflow", "ui", "--host", "0.0.0.0"]
