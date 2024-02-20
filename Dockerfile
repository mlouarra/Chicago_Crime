# Utilisez une image de base officielle de Python
FROM python:3.9

# Définissez le répertoire de travail dans le conteneur
WORKDIR /usr/src/app

# Copiez le contenu du répertoire actuel dans le conteneur à /usr/src/app
COPY . .

# Installez les paquets nécessaires spécifiés dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Rendez le port 8000 disponible pour le monde extérieur à ce conteneur
EXPOSE 8000

# Définissez une variable d'environnement pour le chemin de la base de données
ENV DATABASE_URL /usr/src/app/db/model_evaluation.db

# Lancez main_api.py lorsque le conteneur démarre
CMD ["python", "./main_api.py"]
