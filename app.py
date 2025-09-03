"""QA RAG application with multi-document user sessions."""
import os
import uuid
import pathlib
import datetime
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
import uvicorn
import requests

from sentence_transformers import SentenceTransformer
from utils.loaders import load_source
from utils.splitter import split_text
from utils.exceptions import DocumentLoaderError
from rag_session import RAGSession
from user_session import UserSession

# Load environment variables
load_dotenv()

# --- Configuration ---
SESSION_CLEANUP_INTERVAL_SECONDS = 300
SESSION_TIMEOUT_MINUTES = 15

# --- In-Memory Session Storage ---
sessions: Dict[str, UserSession] = {}

# --- Background Cleanup Logic ---
def _clean_sessions_once():
    now = datetime.datetime.now()
    expiration_time = datetime.timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    expired_ids = [
        session_id for session_id, session in sessions.items()
        if now - session.last_accessed > expiration_time
    ]
    for session_id in expired_ids:
        del sessions[session_id]
        print(f"Cleaned up expired user session: {session_id}")

async def cleanup_expired_sessions_task():
    while True:
        _clean_sessions_once()
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    app.state.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("Embedding model loaded.")

    print("Starting session cleanup task...")
    asyncio.create_task(cleanup_expired_sessions_task())
    yield
    print("Application shutdown.")

# --- App Initialization ---
app = FastAPI(
    title="DocQA",
    description="A RAG application supporting multi-document user sessions.",
    version="2.0.0",
    lifespan=lifespan
)

# --- LLM and CORS Configuration ---
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=GENAI_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.0-flash-lite")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Helper Functions ---
def generate_rag_response(query: str, context_chunks: List[str]) -> str:
    # ... (omitting for brevity, no changes)
    if not context_chunks:
        return "No relevant information found."
    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer the following question based only on the provided context. "
        "If the user asks in a language other than English, respond in their language.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

# --- API Models ---
class SessionResponse(BaseModel):
    session_id: str

class IngestResponse(BaseModel):
    doc_id: str
    source: str

class QueryPayload(BaseModel):
    q: str
    doc_ids: Optional[List[str]] = None

class QuerySource(BaseModel):
    text: str
    score: float
    doc_id: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[QuerySource]

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/sessions", response_model=SessionResponse, summary="Create a new user session")
async def create_session():
    session_id = uuid.uuid4().hex
    sessions[session_id] = UserSession()
    return SessionResponse(session_id=session_id)

@app.post("/sessions/{session_id}/ingest", response_model=IngestResponse, summary="Ingest a document into a session")
async def ingest(session_id: str, request: Request, file: Optional[UploadFile] = File(None), url: Optional[str] = Body(None, embed=True)):
    user_session = sessions.get(session_id)
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")

    # ... (file/url reading logic, omitting for brevity, no changes)
    if not file and not url:
        try:
            body = await request.json()
            if "url" in body:
                url = body["url"]
        except Exception:
            pass
    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide a file or URL.")

    source_name = file.filename if file else url
    content = await file.read() if file else requests.get(url).content
    text = load_source(content, pathlib.Path(source_name).suffix or "url")
    # ... (end of omitted logic)

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text.")

    chunks = split_text(text)
    doc_id = uuid.uuid4().hex

    rag_session = RAGSession(source=source_name, embedding_model=app.state.embedding_model)
    rag_session.ingest(chunks)

    user_session.add_doc(doc_id, rag_session)

    return IngestResponse(doc_id=doc_id, source=source_name)

@app.post("/sessions/{session_id}/query", response_model=QueryResponse, summary="Ask a question within a session")
async def query(session_id: str, payload: QueryPayload):
    user_session = sessions.get(session_id)
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")

    user_session.touch()

    docs_to_query = user_session.get_all_docs()
    if payload.doc_ids:
        docs_to_query = [user_session.get_doc(did) for did in payload.doc_ids if user_session.get_doc(did)]

    all_chunks = []
    for rag_session in docs_to_query:
        # Need to find the doc_id for this rag_session
        doc_id_for_session = next((did for did, rs in user_session.docs.items() if rs == rag_session), None)
        retrieved = rag_session.query(payload.q, k=5)
        for chunk in retrieved:
            chunk['doc_id'] = doc_id_for_session
            chunk['source'] = rag_session.source
        all_chunks.extend(retrieved)

    all_chunks.sort(key=lambda x: x['score'], reverse=True)
    top_chunks = all_chunks[:5]

    relevant_sources = [chunk for chunk in top_chunks if chunk['score'] > 0.5]
    relevant_texts = [chunk['text'] for chunk in relevant_sources]

    answer = generate_rag_response(payload.q, relevant_texts)

    return QueryResponse(answer=answer, sources=relevant_sources)

@app.delete("/sessions/{session_id}/documents/{doc_id}", status_code=204, summary="Delete a document from a session")
async def delete_document(session_id: str, doc_id: str):
    """Deletes a specific document from a user session."""
    user_session = sessions.get(session_id)
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")

    if not user_session.get_doc(doc_id):
        raise HTTPException(status_code=404, detail="Document not found in this session.")

    user_session.remove_doc(doc_id)
    return

# ... (Main Execution block, no changes)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
