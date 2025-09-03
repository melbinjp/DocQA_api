import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os
import requests
import datetime

# Set a dummy API key for tests
os.environ['GOOGLE_API_KEY'] = 'test-key'

# Mock asyncio.create_task BEFORE the app is imported to prevent the
# background task from starting during tests.
with patch('asyncio.create_task'):
    from app import app, sessions, _clean_sessions_once
    from rag_session import RAGSession

# Use a client that can be reset for each test
@pytest.fixture
def client():
    sessions.clear()
    yield TestClient(app)
    sessions.clear()

def test_health_check(client):
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["active_sessions"] == 0

def test_query_single_document(client, mocker):
    """Tests a full cycle with a single document ID."""
    mock_llm_call = mocker.patch("app.llm_model.generate_content")
    mock_llm_call.return_value.text = "The answer is based on the cat document."

    # 1. Ingest
    response_ingest = client.post("/ingest", files={"file": ("cat_doc.txt", b"about cats", "text/plain")})
    assert response_ingest.status_code == 200
    doc_id = response_ingest.json()["doc_id"]

    # 2. Mock session query
    mocker.patch.object(sessions[doc_id], 'query', return_value=[{"text": "about cats", "score": 0.9}])

    # 3. Query using the unified endpoint with a single ID
    response_query = client.post("/query", json={"doc_ids": doc_id, "q": "What about cats?"})
    assert response_query.status_code == 200

    data = response_query.json()
    assert data["answer"] == "The answer is based on the cat document."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "cat_doc.txt"

def test_query_multiple_documents(client, mocker):
    """Tests querying across two different documents with the unified endpoint."""
    mock_llm_call = mocker.patch("app.llm_model.generate_content")
    mock_llm_call.return_value.text = "Combined answer."

    # Ingest docs
    resp_a = client.post("/ingest", files={"file": ("doc_a.txt", b"feline behavior", "text/plain")})
    doc_id_a = resp_a.json()["doc_id"]
    resp_b = client.post("/ingest", files={"file": ("doc_b.txt", b"canine training", "text/plain")})
    doc_id_b = resp_b.json()["doc_id"]

    # Mock query responses
    mocker.patch.object(sessions[doc_id_a], 'query', return_value=[{"text": "feline chunk", "score": 0.9}])
    mocker.patch.object(sessions[doc_id_b], 'query', return_value=[{"text": "canine chunk", "score": 0.8}])

    # Query both documents
    response_query = client.post("/query", json={"doc_ids": [doc_id_a, doc_id_b], "q": "What do you know?"})
    assert response_query.status_code == 200

    data = response_query.json()
    assert data["answer"] == "Combined answer."
    assert len(data["sources"]) == 2

    # Check that sources are sorted by score and contain correct data
    assert data["sources"][0]["text"] == "feline chunk"
    assert data["sources"][0]["source"] == "doc_a.txt"
    assert data["sources"][0]["doc_id"] == doc_id_a

    assert data["sources"][1]["text"] == "canine chunk"
    assert data["sources"][1]["source"] == "doc_b.txt"
    assert data["sources"][1]["doc_id"] == doc_id_b

def test_query_nonexistent_doc_id(client):
    """Tests querying with a doc_id that does not exist."""
    # Note: The current logic doesn't raise an error for non-existent IDs, it just returns no results.
    # This is a valid design choice.
    response = client.post("/query", json={"doc_ids": ["fake-id-123"], "q": "Any question"})
    assert response.status_code == 200
    assert len(response.json()["sources"]) == 0
    assert "No relevant information found" in response.json()["answer"]

def test_ingest_url_error(client, mocker):
    """Tests that a failure during URL fetching is handled gracefully."""
    mocker.patch("app.requests.get", side_effect=requests.RequestException("Connection failed"))
    response = client.post("/ingest", json={"url": "http://example.com/bad.url"})
    assert response.status_code == 400
    assert "Error fetching URL" in response.json()["detail"]

def test_session_cleanup_logic():
    """Tests the single-pass cleanup logic directly."""
    sessions.clear()

    # Create sessions with sources
    fresh_session = RAGSession(source="fresh.txt")
    sessions["fresh_doc"] = fresh_session

    expired_session = RAGSession(source="expired.txt")
    expired_session.last_accessed = datetime.datetime.now() - datetime.timedelta(minutes=20)
    sessions["expired_doc"] = expired_session

    assert "fresh_doc" in sessions
    assert "expired_doc" in sessions

    _clean_sessions_once()

    assert "fresh_doc" in sessions
    assert "expired_doc" not in sessions

    sessions.clear()
