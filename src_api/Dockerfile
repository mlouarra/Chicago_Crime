# Utilisation d'une image de base Python
FROM python:3.9-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Copie des fichiers requis dans le conteneur
COPY ./src_api /app

# Installation des dépendances
RUN pip install fastapi uvicorn

# Exposition du port utilisé par l'application FastAPI
EXPOSE 8000

# Commande pour démarrer l'application avec uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
