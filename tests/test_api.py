from app.main import app
from fastapi.testclient import TestClient
import sys
import os

# Add project root to sys.path BEFORE importing app.main
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))


client = TestClient(app)


def test_predict():
    response = client.post("/predict", json={"prices": [0.1]*5})
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert isinstance(response.json()["predicted_price"], float)
