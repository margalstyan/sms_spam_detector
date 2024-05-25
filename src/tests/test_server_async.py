import pytest
import pytest_asyncio
import httpx
from httpx import ASGITransport

from src.server.main import app


# Fixture for creating an async test client
@pytest_asyncio.fixture
async def async_client():
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        yield client


# Parametrized test cases for the /info endpoint
@pytest.mark.parametrize("url, status_code, response_keys", [
    ("/info", 200, ["algorithm_name", "related_research_papers", "version", "training_data"]),
])
@pytest.mark.asyncio
async def test_info_endpoint(async_client, url, status_code, response_keys):
    response = await async_client.get(url)
    assert response.status_code == status_code
    for key in response_keys:
        assert key in response.json()


# Parametrized test cases for the /predict endpoint
@pytest.mark.parametrize("message, expected_status_code, expected_prediction", [
    ("Click on link below to earn $5000", 200, "spam"),
    ("Hello, how are you?", 200, "ham"),
    ("", 400, "Message should not be empty"),
    (" ", 400, "Message should not be empty"),
])
@pytest.mark.asyncio
async def test_predict_endpoint(async_client, message, expected_status_code, expected_prediction):
    response = await async_client.post("/predict", json={"message": message})
    assert response.status_code == expected_status_code
    if response.status_code == 200:
        assert response.json()["prediction"] == expected_prediction
    else:
        assert response.json()["detail"] == expected_prediction


@pytest.mark.asyncio
async def test_middleware(async_client):
    response = await async_client.get("/")
    assert "swagger-ui" in str(response.content)
