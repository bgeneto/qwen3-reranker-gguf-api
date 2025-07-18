import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app, get_llm


@pytest.fixture(scope="module")
def client():
    # Mock the model loading
    with patch("app.main.Llama") as mock_llama:
        mock_llama.return_value = MagicMock()
        get_llm() # Ensure the mock is set
        yield TestClient(app)

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rerank_requests_total" in response.text

def test_rerank_unauthorized(client):
    response = client.post("/v1/rerank", json={
        "query": "test",
        "documents": []
    })
    assert response.status_code == 401

@patch("app.main.rerank_one")
def test_rerank_success(mock_rerank_one, client):
    mock_rerank_one.return_value = 0.9
    
    response = client.post(
        "/v1/rerank",
        headers={"Authorization": "Bearer change-me-please"},
        json={
            "query": "What is the capital of China?",
            "documents": [{"text": "The capital of China is Beijing."}]
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "qwen3-reranker"
    assert len(data["results"]) == 1
    assert data["results"][0]["relevance_score"] == 0.9
