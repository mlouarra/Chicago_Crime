from fastapi.testclient import TestClient
from src.auth_api import app # Remplacez 'my_app' par le nom de votre application FastAPI
import jwt
from datetime import datetime, timedelta

# Clé secrète (la même que celle utilisée dans votre application)
SECRET_KEY = "my_secret_key"
# Algorithme de chiffrement (le même que celui utilisé dans votre application)
ALGORITHM = "HS256"
# Durée de validité du token
ACCESS_TOKEN_EXPIRE_MINUTES = 30

fake_users_db = {
    "chicago_user": {
        "username": "chicago_user",
        "hashed_password": "$2b$12$yBwveOeJZ3qqd4RB0j1VwO/YhUC8tTdPxXADtSloV8P.s62Ucu3ZC",
    }
}
# Fonction pour créer un token JWT valide
def create_valid_token(username):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "exp": expire,
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    print(f"Created a valid JWT token for user: {token}")
    return token

client = TestClient(app)

def test_get_current_user_success():
    # Simulez un token JWT valide
    valid_token = create_valid_token("chicago_user")

    # Effectuez une requête à votre endpoint FastAPI
    response = client.get("/secure-data", headers={"Authorization": f"Bearer {valid_token}"})

    # Vérifiez le code de statut de la réponse
    assert response.status_code == 200

    # Vérifiez le contenu de la réponse
    response_json = response.json()
    assert "message" in response_json
    assert response_json["message"] == "Operation successful"
    assert "user" in response_json

def test_get_current_user_failure():
    # Simulez un token JWT invalide
    invalid_token = "token_jwt_invalide"

    # Effectuez une requête à votre endpoint FastAPI
    response = client.get("/secure-data", headers={"Authorization": f"Bearer {invalid_token}"})

    # Vérifiez le code de statut de la réponse (401 Unauthorized)
    assert response.status_code

    # Vérifiez le contenu de la réponse (message d'erreur)
    response_json = response.json()
    assert "detail" in response_json
    assert response_json["detail"] == "Could not validate credentials"
