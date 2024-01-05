import passlib
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional
import jwt
import datetime
import secrets

app = FastAPI()

# Configuration secrète pour la génération du token JWT
SECRET_KEY = secrets.token_hex(32)
ALGORITHM = 'HS256'

# Utilisateur et mot de passe stockés
username = "chicago"
password = "password"

# Création d'une instance CryptContext pour hacher les mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Générer le hachage du mot de passe en utilisant la fonction generate_password_hash de Passlib
hashed_password = pwd_context.hash(password)

# schéma OAuth2 qui sera utilisé pour valider les jetons d'accès lorsqu'ils sont envoyés dans les requêtes HTTP.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Fonction pour vérifier le mot de passe haché
def verify_password(password, hashed_password):
    """
        Vérifie si le mot de passe fourni correspond au hachage du mot de passe stocké.

        Args:
            password (str): Mot de passe à vérifier.
            hashed_password (str): Hachage du mot de passe stocké.

        Returns:
            bool: True si le mot de passe correspond, sinon False.
        """
    return pwd_context.verify(password, hashed_password)

# Fonction pour générer un token JWT
def generate_token(username):
    """
        Génère un token JWT pour un utilisateur donné avec une expiration de 30 minutes.

        Args:
            username (str): Le nom d'utilisateur pour lequel le token est généré.

        Returns:
            str: Le token JWT généré.
        """
    expiration_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    payload = {
        'exp': expiration_time,  # Durée de validité du token (30 minutes)
        'iat': datetime.datetime.utcnow(),
        'sub': username  # Sujet du token
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

def get_token(authorization: str = Header(None)) -> str:
    """
        Extrait le token JWT du champ d'autorisation dans l'en-tête HTTP.

        Args:
            authorization (str, optional): En-tête d'autorisation HTTP contenant le token JWT.

        Returns:
            str: Le token JWT extrait de l'en-tête d'autorisation.

        Raises:
            HTTPException: Si le token est manquant ou invalide.
        """
    if authorization is None:
        raise HTTPException(status_code=401, detail='Token non fourni')
    token_prefix, token = authorization.split()

    if token_prefix.lower() != 'bearer':
        raise HTTPException(status_code=401, detail='Token invalide')
    return token

def get_current_user(token: str = Depends(oauth2_scheme)):
    """
        Obtient l'utilisateur actuel à partir du token JWT.

        Args:
            token (str): Le token JWT d'autorisation.

        Returns:
            UserLogin: Un objet UserLogin contenant le nom d'utilisateur et le token JWT.

        Raises:
            HTTPException: Si le token JWT est expiré ou invalide.
        """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        current_user = payload['sub']
        return UserLogin(username=current_user, token=token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail='Token expiré')
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail='Token invalide')

# Modèle de données Pydantic pour les informations d'authentification
class UserLogin(BaseModel):
    username: str
    password: str
    token: Optional[str] = None  # Champ optionnel pour stocker le token JWT
'''
# Route d'authentification
@app.post('/login', response_model=dict, tags=["Authentification"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        if form_data.username == username and verify_password(form_data.password, hashed_password):
            token = generate_token(username)
            return {'access_token': token, 'token_type': 'bearer'}
        else:
            raise HTTPException(status_code=401, detail='Échec de l\'authentification')
    except passlib.exc.UnknownHashError:
        raise HTTPException(status_code=500, detail='Erreur interne du serveur : Format de hachage inconnu')

# Route sécurisée nécessitant un token
@app.get('/secure-data', response_model=dict, tags=["Données sécurisées"])
async def secure_data(current_user: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(current_user, SECRET_KEY, algorithms=[ALGORITHM])
        current_username = payload['sub']
        return {'message': f'Données sécurisées pour l\'utilisateur {current_username}'}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail='Token expiré')
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail='Token invalide')
'''
