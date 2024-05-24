import pytest
from fastapi.testclient import TestClient
from src.server.main import app


# Fixture to create a test client
@pytest.fixture
def client():
    return TestClient(app)


# Parametrized test cases for the /info endpoint
@pytest.mark.parametrize("url, status_code, response_keys", [
    ("/info", 200, ["algorithm_name", "related_research_papers", "version", "training_data"]),
])
def test_info_endpoint(client, url, status_code, response_keys):
    response = client.get(url)
    assert response.status_code == status_code
    for key in response_keys:
        assert key in response.json()


# Parametrized test cases for the /predict endpoint
@pytest.mark.parametrize("message, expected_status_code, expected_prediction", [
    ("Click on link below to earn $5000", 200, "spam"),
    ("Hello, how are you?", 200, "ham"),
    ("", 400, "Message should not be empty"),
])
def test_predict_endpoint(client, message, expected_status_code, expected_prediction):
    response = client.post("/predict", json={"message": message})
    assert response.status_code == expected_status_code
    if response.status_code == 200:
        assert response.json()["prediction"] == expected_prediction
    else:
        assert response.json()["detail"] == expected_prediction
