import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os
import requests

# Set a dummy API key for tests
os.environ['GOOGLE_API_KEY'] = 'test-key'

from app import app, sessions

# Use a client that can be reset for each test
@pytest.fixture
def client():
    # Clear any existing sessions before each test run
    sessions.clear()
    yield TestClient(app)
    sessions.clear()

def test_health_check(client):
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "ok"
    assert json_response["active_sessions"] == 0

def test_ingest_file_and_query_success(client, mocker):
    """Tests a full successful cycle: ingest a file, then query it."""
    # Mock the LLM call to isolate the test from the external API
    mock_llm_call = mocker.patch("app.llm_model.generate_content")
    mock_llm_call.return_value.text = "The answer is based on the test document."

    # 1. Ingest a document
    file_content = b"This is a test document about cats."
    response_ingest = client.post(
        "/ingest",
        files={"file": ("cat_document.txt", file_content, "text/plain")}
    )
    assert response_ingest.status_code == 200
    ingest_data = response_ingest.json()
    doc_id = ingest_data.get("doc_id")
    assert doc_id is not None
    assert ingest_data["chunks_ingested"] > 0
    assert len(sessions) == 1

    # Mock the query method on the specific session instance
    mocker.patch.object(
        sessions[doc_id],
        'query',
        return_value=[{"text": "This is a test document about cats.", "score": 0.99}]
    )

    # 2. Query the ingested document
    response_query = client.post(
        "/query",
        json={"doc_id": doc_id, "q": "What is this document about?"}
    )
    assert response_query.status_code == 200
    query_data = response_query.json()

    # Assert that the pipeline ran correctly with the mocked data
    assert len(query_data["sources"]) == 1
    assert query_data["sources"][0]["text"] == "This is a test document about cats."
    mock_llm_call.assert_called_once()
    assert query_data["answer"] == "The answer is based on the test document."

def test_query_nonexistent_doc_id(client):
    """Tests querying with a doc_id that does not exist."""
    response = client.post(
        "/query",
        json={"doc_id": "fake-id-123", "q": "Any question"}
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]

def test_session_isolation(client, mocker):
    """
    Tests that two different document sessions are completely isolated.
    """
    mock_llm_call = mocker.patch("app.llm_model.generate_content")
    mock_llm_call.return_value.text = "This is a generic answer."

    # Ingest document A (about cats)
    content_a = b"A document discussing feline behavior."
    response_a = client.post("/ingest", files={"file": ("doc_a.txt", content_a, "text/plain")})
    doc_id_a = response_a.json()["doc_id"]

    # Ingest document B (about dogs)
    content_b = b"An article about canine training."
    response_b = client.post("/ingest", files={"file": ("doc_b.txt", content_b, "text/plain")})
    doc_id_b = response_b.json()["doc_id"]

    assert len(sessions) == 2
    assert doc_id_a != doc_id_b

    # Query document B with a question only document A can answer
    response_query = client.post(
        "/query",
        json={"doc_id": doc_id_b, "q": "What is feline behavior?"}
    )

    assert response_query.status_code == 200
    query_data = response_query.json()

    # The key assertion: no sources should be found from doc B for a question about doc A
    assert len(query_data["sources"]) == 0
    # The LLM should then generate a response saying it doesn't know
    assert "No relevant information found" in query_data["answer"]

    # Verify the LLM was NOT called, because no relevant context was found
    mock_llm_call.assert_not_called()

def test_ingest_url_error(client, mocker):
    """Tests that a failure during URL fetching is handled gracefully."""
    mock_requests_get = mocker.patch("app.requests.get")
    mock_requests_get.side_effect = requests.RequestException("Connection failed")

    response = client.post("/ingest", json={"url": "http://example.com/bad.url"})

    assert response.status_code == 400
    assert "Error fetching URL" in response.json()["detail"]
