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
with patch('asyncio.create_task') as mock_create_task:
    from app import app, sessions, _clean_sessions_once
    from rag_session import RAGSession

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
    """Tests that two different document sessions are completely isolated."""
    mock_llm_call = mocker.patch("app.llm_model.generate_content")
    mock_llm_call.return_value.text = "This is a generic answer."

    content_a = b"A document discussing feline behavior."
    response_a = client.post("/ingest", files={"file": ("doc_a.txt", content_a, "text/plain")})
    doc_id_a = response_a.json()["doc_id"]

    content_b = b"An article about canine training."
    response_b = client.post("/ingest", files={"file": ("doc_b.txt", content_b, "text/plain")})
    doc_id_b = response_b.json()["doc_id"]

    mocker.patch.object(sessions[doc_id_b], 'query', return_value=[])

    response_query = client.post(
        "/query",
        json={"doc_id": doc_id_b, "q": "What is feline behavior?"}
    )

    assert response_query.status_code == 200
    query_data = response_query.json()

    assert len(query_data["sources"]) == 0
    assert "No relevant information found" in query_data["answer"]
    mock_llm_call.assert_not_called()

def test_ingest_url_error(client, mocker):
    """Tests that a failure during URL fetching is handled gracefully."""
    mock_requests_get = mocker.patch("app.requests.get")
    mock_requests_get.side_effect = requests.RequestException("Connection failed")

    response = client.post("/ingest", json={"url": "http://example.com/bad.url"})

    assert response.status_code == 400
    assert "Error fetching URL" in response.json()["detail"]

def test_session_cleanup_logic():
    """Tests the single-pass cleanup logic directly."""
    # Clear sessions to ensure a clean state for this test
    sessions.clear()

    # 1. Create a fresh session
    fresh_session = RAGSession()
    sessions["fresh_doc"] = fresh_session

    # 2. Create an expired session
    expired_session = RAGSession()
    expired_session.last_accessed = datetime.datetime.now() - datetime.timedelta(minutes=20)
    sessions["expired_doc"] = expired_session

    assert "fresh_doc" in sessions
    assert "expired_doc" in sessions

    # 3. Run the cleanup logic
    _clean_sessions_once()

    # 4. Assert that only the expired session was removed
    assert "fresh_doc" in sessions
    assert "expired_doc" not in sessions

    # Final cleanup
    sessions.clear()

def test_query_multiple_documents(client, mocker):
    """Tests querying across two different documents."""
    mock_llm_call = mocker.patch("app.llm_model.generate_content")
    mock_llm_call.return_value.text = "Combined answer."

    # Ingest document A (about cats)
    content_a = b"A document discussing feline behavior."
    response_a = client.post("/ingest", files={"file": ("doc_a.txt", content_a, "text/plain")})
    doc_id_a = response_a.json()["doc_id"]

    # Ingest document B (about dogs)
    content_b = b"An article about canine training."
    response_b = client.post("/ingest", files={"file": ("doc_b.txt", content_b, "text/plain")})
    doc_id_b = response_b.json()["doc_id"]

    # Mock the query responses for each session
    mocker.patch.object(
        sessions[doc_id_a],
        'query',
        return_value=[{"text": "High score cat chunk", "score": 0.9}]
    )
    mocker.patch.object(
        sessions[doc_id_b],
        'query',
        return_value=[{"text": "Low score dog chunk", "score": 0.6}]
    )

    # Query both documents
    response_query = client.post(
        "/query-multiple",
        json={"doc_ids": [doc_id_a, doc_id_b], "q": "What do you know?"}
    )

    assert response_query.status_code == 200
    query_data = response_query.json()

    # Check that the answer is based on the combined context
    assert query_data["answer"] == "Combined answer."

    # Check that the sources include the high-scoring chunk and the correct doc_id
    assert len(query_data["sources"]) == 2
    assert query_data["sources"][0]["text"] == "High score cat chunk"
    assert query_data["sources"][0]["doc_id"] == doc_id_a
    assert query_data["sources"][1]["text"] == "Low score dog chunk"
    assert query_data["sources"][1]["doc_id"] == doc_id_b

    # Check that the LLM was called with the right context
    mock_llm_call.assert_called_once()
    prompt_arg = mock_llm_call.call_args[0][0]
    assert "High score cat chunk" in prompt_arg
    assert "Low score dog chunk" in prompt_arg
