import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os
import datetime

# Set a dummy API key for tests
os.environ['GOOGLE_API_KEY'] = 'test-key'

# Mock asyncio.create_task BEFORE the app is imported
with patch('asyncio.create_task'):
    from app import app, sessions, _clean_sessions_once
    from user_session import UserSession
    from rag_session import RAGSession

# Use a client that handles the lifespan context
@pytest.fixture
def client():
    sessions.clear()
    # Using the 'with' statement ensures that startup and shutdown events are run
    with TestClient(app) as test_client:
        yield test_client
    sessions.clear()

def test_create_session(client):
    """Tests that a new user session can be created."""
    response = client.post("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["session_id"] in sessions
    assert isinstance(sessions[data["session_id"]], UserSession)

def test_ingest_into_session(client, mocker):
    """Tests ingesting a document into a created session."""
    session_id = client.post("/sessions").json()["session_id"]

    mocker.patch("app.load_source", return_value="Test content")
    response = client.post(
        f"/sessions/{session_id}/ingest",
        files={"file": ("test.txt", b"...", "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert "num_chunks" in data
    assert data["num_chunks"] == 1  # "Test content" should be a single chunk

    user_session = sessions[session_id]
    assert len(user_session.docs) == 1
    doc_id = data["doc_id"]
    assert user_session.get_doc(doc_id) is not None

def test_query_session(client, mocker):
    """Tests querying documents within a session."""
    session_id = client.post("/sessions").json()["session_id"]
    mocker.patch("app.load_source", return_value="Content A")
    resp_a = client.post(f"/sessions/{session_id}/ingest", files={"file": ("doc_a.txt", b"A", "text/plain")})
    doc_id_a = resp_a.json()["doc_id"]

    mocker.patch("app.load_source", return_value="Content B")
    resp_b = client.post(f"/sessions/{session_id}/ingest", files={"file": ("doc_b.txt", b"B", "text/plain")})
    doc_id_b = resp_b.json()["doc_id"]

    user_session = sessions[session_id]
    mocker.patch.object(user_session.get_doc(doc_id_a), 'query', return_value=[{"text": "from A", "score": 0.9}])
    mocker.patch.object(user_session.get_doc(doc_id_b), 'query', return_value=[{"text": "from B", "score": 0.8}])

    # Mock the async generator
    async def mock_async_gen(*args, **kwargs):
        yield "Final Answer"
    mocker.patch("app.generate_rag_response", side_effect=mock_async_gen)

    # Query all docs in session
    response_all = client.post(f"/sessions/{session_id}/query", json={"q": "test"})
    assert response_all.status_code == 200
    assert response_all.json()["answer"] == "Final Answer"

    # Query a specific doc in session
    response_specific = client.post(f"/sessions/{session_id}/query", json={"q": "test", "doc_ids": [doc_id_a]})
    assert response_specific.status_code == 200
    assert response_specific.json()["answer"] == "Final Answer"

def test_delete_document_from_session(client, mocker):
    """Tests deleting a document from a session."""
    session_id = client.post("/sessions").json()["session_id"]
    mocker.patch("app.load_source", return_value="Test content")
    resp = client.post(f"/sessions/{session_id}/ingest", files={"file": ("test.txt", b"...", "text/plain")})
    doc_id = resp.json()["doc_id"]

    assert len(sessions[session_id].docs) == 1

    response_delete = client.delete(f"/sessions/{session_id}/documents/{doc_id}")
    assert response_delete.status_code == 204

    assert len(sessions[session_id].docs) == 0

def test_session_cleanup_logic():
    """Tests the single-pass cleanup logic directly."""
    sessions.clear()

    # We need the model to instantiate the RAGSession
    # In a real test setup, we might mock this, but for now, we rely on the app state
    # This test must run within a context where the app lifespan has started.
    # For direct calling, we can manually set it if needed, or rely on other tests
    # having populated it. Let's ensure it's there.
    if not hasattr(app.state, "embedding_model"):
        from sentence_transformers import SentenceTransformer
        app.state.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    fresh_session = UserSession()
    fresh_session.add_doc("doc1", RAGSession(source="fresh.txt", embedding_model=app.state.embedding_model))
    sessions["fresh_session"] = fresh_session

    expired_session = UserSession()
    expired_session.add_doc("doc2", RAGSession(source="expired.txt", embedding_model=app.state.embedding_model))
    expired_session.last_accessed = datetime.datetime.now() - datetime.timedelta(minutes=20)
    sessions["expired_session"] = expired_session

    assert "fresh_session" in sessions
    assert "expired_session" in sessions

    _clean_sessions_once()

    assert "fresh_session" in sessions
    assert "expired_session" not in sessions

    sessions.clear()
