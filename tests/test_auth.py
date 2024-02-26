import pytest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from fastapi.testclient import TestClient
from main_api import (app, verify_password, generate_token)

# Test la fonction verify_password
def test_verify_password():
    hashed_password = "$2b$12$x518gVtvF9nE4RDDInHRIe07GBDfXkl9PwMId3QolhAVeG5fYTuIy"
    assert verify_password("password", hashed_password) is True
    assert verify_password("wrong_password", hashed_password) is False

# Test la génération de token
def test_generate_token():
    username = "test_user"
    token = generate_token(username)
    assert isinstance(token, str)
    assert len(token) > 0

# Test de la route d'authentification
client = TestClient(app)

def test_login_success():
    response = client.post("/login", data={"username": "chicago", "password": "password"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_failure():
    response = client.post("/login", data={"username": "chicago", "password": "wrong_password"})
    assert response.status_code == 401
    assert "access_token" not in response.json()

# Test de la route de données sécurisées
def test_secure_data_with_token():
    token = generate_token("chicago")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/secure-data", headers=headers)
    assert response.status_code == 200
    assert response.json()["message"] == "Données sécurisées pour l'utilisateur chicago"

def test_secure_data_without_token():
    response = client.get("/secure-data")
    assert response.status_code == 401
