"""
Integration tests for the DocQA API.

These tests are designed to validate the end-to-end functionality of the API,
treating the application as a black box. They do not mock the core components
like the RAG session or embedding models, ensuring that the entire pipeline
is tested.
"""
import pytest
from fastapi.testclient import TestClient
import os

# Set a dummy API key for tests if not already set
os.environ['GOOGLE_API_KEY'] = os.environ.get('GOOGLE_API_KEY', 'test-key')

from app import app, sessions

@pytest.fixture
def client():
    """Provides a TestClient that handles the app's lifespan."""
    sessions.clear()
    with TestClient(app) as test_client:
        yield test_client
    sessions.clear()

from unittest.mock import patch, AsyncMock, MagicMock
import json

def test_multi_document_query_correctness(client):
    """
    Tests that a query across multiple documents returns the most relevant
    result from the correct source document. This tests the non-streaming path.
    """
    session_id = client.post("/sessions").json()["session_id"]

    doc_a_content = "The sky is blue and clouds are white."
    doc_b_content = "The grass is green and the soil is brown."

    client.post(f"/sessions/{session_id}/ingest", files={"file": ("doc_a.txt", doc_a_content.encode("utf-8"))})
    client.post(f"/sessions/{session_id}/ingest", files={"file": ("doc_b.txt", doc_b_content.encode("utf-8"))})

    # Mock the non-streaming response from the LLM helper
    mock_answer = "The grass is indeed green."
    async def mock_async_gen():
        yield mock_answer

    with patch("app.generate_rag_response", return_value=mock_async_gen()) as mock_llm_call:
        response = client.post(
            f"/sessions/{session_id}/query",
            json={"q": "What color is the grass?", "stream": False}
        )

    assert response.status_code == 200
    data = response.json()
    sources = data["sources"]

    assert data["answer"] == mock_answer
    assert len(sources) > 0, "Query should return at least one source."
    assert sources[0]["source"] == "doc_b.txt"
    assert "grass is green" in sources[0]["text"]

def test_ingest_from_url(client):
    """
    Tests that a document can be ingested from a URL. Mocks the network call.
    """
    session_id = client.post("/sessions").json()["session_id"]
    test_url = "http://example.com/test_document.txt"
    test_content = b"This is the content from a URL."

    from app import app
    with patch.object(app.state.http_client, 'get', new_callable=AsyncMock) as mock_async_get:
        mock_response = mock_async_get.return_value
        mock_response.status_code = 200
        mock_response.content = test_content
        mock_response.raise_for_status.return_value = None

        response = client.post(f"/sessions/{session_id}/ingest", json={"url": test_url})

    assert response.status_code == 200, f"API returned error: {response.text}"
    data = response.json()
    assert data["source"] == test_url

def test_ingestion_uses_cache(client):
    """
    Tests that the ingestion process uses the session cache to avoid
    re-embedding identical chunks.
    """
    session_id = client.post("/sessions").json()["session_id"]

    shared_content = "This is a shared document chunk that should be cached."
    new_content = "This is a new document chunk that should be encoded."

    # Ingest the shared content to populate the cache
    response1 = client.post(f"/sessions/{session_id}/ingest", files={"file": ("shared.txt", shared_content.encode("utf-8"))})
    assert response1.status_code == 200

    # Ingest new content to ensure the mock works
    response2 = client.post(f"/sessions/{session_id}/ingest", files={"file": ("new.txt", new_content.encode("utf-8"))})
    assert response2.status_code == 200

    from app import app
    # Patch the encode method to monitor its calls
    with patch.object(app.state.embedding_model, 'encode', wraps=app.state.embedding_model.encode) as mock_encode:
        # Re-ingest the shared content
        response3 = client.post(f"/sessions/{session_id}/ingest", files={"file": ("shared_again.txt", shared_content.encode("utf-8"))})
        assert response3.status_code == 200

        # Assert that the encode function was NOT called, because the chunk was cached
        mock_encode.assert_not_called()

def test_query_streaming_response(client):
    """Tests that the query endpoint returns a valid SSE stream when requested."""
    session_id = client.post("/sessions").json()["session_id"]
    client.post(f"/sessions/{session_id}/ingest", files={"file": ("doc.txt", b"The answer is 42.", "text/plain")})

    mock_tokens = ["The", " answer", " is", " 42", "."]

    async def mock_stream_generator():
        for token in mock_tokens:
            yield f"data: {json.dumps({'token': token})}\n\n"

    with patch("app.generate_rag_response", return_value=mock_stream_generator()) as mock_llm:
        response = client.post(
            f"/sessions/{session_id}/query",
            json={"q": "What is the answer?", "stream": True},
            timeout=10,
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    stream_content = response.text

    # Reconstruct the answer from tokens to verify all were received
    received_tokens = [json.loads(line.split("data: ")[1]) for line in stream_content.strip().split('\n\n') if "token" in line]
    reconstructed_answer = "".join([item['token'] for item in received_tokens])
    assert reconstructed_answer == "".join(mock_tokens)

    # Check that the sources and end events are also present
    assert '{"type": "sources"' in stream_content
    assert '{"type": "end"}' in stream_content
