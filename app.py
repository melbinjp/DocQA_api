"""QA RAG application for stateless, session-based document questioning."""
import os
import uuid
import pathlib
import datetime
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
import uvicorn
import requests

# Import utilities and the session manager
from utils.loaders import load_source
from utils.splitter import split_text
from utils.exceptions import DocumentLoaderError
from rag_session import RAGSession

# Load environment variables
load_dotenv()

# --- Configuration ---
SESSION_CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes
SESSION_TIMEOUT_MINUTES = 15

# --- In-Memory Session Storage ---
sessions: Dict[str, RAGSession] = {}

# --- Background Cleanup Logic (Refactored for Testability) ---
def _clean_sessions_once():
    """Single pass to find and remove expired sessions."""
    now = datetime.datetime.now()
    expiration_time = datetime.timedelta(minutes=SESSION_TIMEOUT_MINUTES)

    expired_sessions = [
        doc_id for doc_id, session in sessions.items()
        if now - session.last_accessed > expiration_time
    ]

    for doc_id in expired_sessions:
        del sessions[doc_id]
        print(f"Cleaned up expired session: {doc_id}")

async def cleanup_expired_sessions_task():
    """The background task that runs periodically."""
    while True:
        _clean_sessions_once()
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup: Starting session cleanup task...")
    asyncio.create_task(cleanup_expired_sessions_task())
    yield
    print("Application shutdown.")

# --- App and Model Initialization ---
app = FastAPI(
    title="DocQA",
    description="An application for asking questions about documents using a stateless, session-based RAG architecture with session timeouts.",
    version="1.1.0",
    lifespan=lifespan
)

# Configure the Generative AI model
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=GENAI_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.0-flash-lite")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def generate_rag_response(query: str, context_chunks: List[str]) -> str:
    """Generates a response from the LLM based on the retrieved context."""
    if not context_chunks:
        return "No relevant information found in the document to answer this question."

    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant. Answer the following question based *only* on the provided context. "
        "If the answer is not available in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate a response from the language model.")

# --- API Models ---
class IngestResponse(BaseModel):
    doc_id: str = Field(..., description="The unique ID for the ingested document session.")
    source: str = Field(..., description="The original source of the document (filename or URL).")
    chunks_ingested: int = Field(..., description="The number of text chunks the document was split into.")

class QueryPayload(BaseModel):
    doc_id: str = Field(..., description="The ID of the document session to query against.")
    q: str = Field(..., description="The question to ask.")

class QuerySource(BaseModel):
    text: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[QuerySource]

class QueryMultiplePayload(BaseModel):
    doc_ids: List[str] = Field(..., description="A list of document session IDs to query against.")
    q: str = Field(..., description="The question to ask.")

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health", summary="Health check")
async def health():
    """Provides the status of the service and the number of active document sessions."""
    return {
        "status": "ok",
        "active_sessions": len(sessions)
    }

@app.post("/ingest", response_model=IngestResponse, summary="Ingest a document")
async def ingest(
    request: Request,
    file: Optional[UploadFile] = File(None, description="A document file to ingest."),
    url: Optional[str] = Body(None, description="URL of a document to ingest.", embed=True)
) -> IngestResponse:
    if not file and not url:
        try:
            body = await request.json()
            if "url" in body:
                url = body["url"]
        except Exception:
            pass

    if not file and not url:
        raise HTTPException(status_code=400, detail="You must provide either a file or a URL.")

    doc_id = uuid.uuid4().hex
    source_name = ""
    content = b""

    try:
        if file:
            source_name = file.filename
            content = await file.read()
            file_ext = pathlib.Path(source_name).suffix
            text = load_source(content, file_ext)
        elif url:
            source_name = url
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, timeout=15, headers=headers)
            resp.raise_for_status()
            content = resp.content
            text = load_source(content, "url")
    except DocumentLoaderError as e:
        raise HTTPException(status_code=400, detail=f"Failed to load document: {e}")
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during ingestion: {e}")

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail=f"Could not extract any text from the source: {source_name}")

    chunks = split_text(text, max_chars=512, overlap=100)
    if not chunks:
        raise HTTPException(status_code=400, detail=f"Failed to split the document into chunks: {source_name}")

    session = RAGSession()
    session.ingest(chunks)
    sessions[doc_id] = session

    return IngestResponse(doc_id=doc_id, source=source_name, chunks_ingested=len(chunks))

@app.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query(payload: QueryPayload) -> QueryResponse:
    session = sessions.get(payload.doc_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Document session with doc_id '{payload.doc_id}' not found.")

    session.touch()

    retrieved_chunks = session.query(payload.q, k=5)

    relevant_texts = [chunk['text'] for chunk in retrieved_chunks if chunk['score'] > 0.5]
    answer = generate_rag_response(payload.q, relevant_texts)

    relevant_sources = [chunk for chunk in retrieved_chunks if chunk['score'] > 0.5]

    return QueryResponse(answer=answer, sources=relevant_sources)


@app.post("/query-multiple", response_model=QueryResponse, summary="Ask a question across multiple documents")
async def query_multiple(payload: QueryMultiplePayload) -> QueryResponse:
    """
    Asks a question against a list of specified document sessions.
    This allows for searching across multiple documents at once.
    """
    all_chunks = []
    for doc_id in payload.doc_ids:
        session = sessions.get(doc_id)
        if session:
            session.touch()
            # We can add the doc_id to the source for better traceability
            retrieved_chunks = session.query(payload.q, k=5)
            for chunk in retrieved_chunks:
                chunk['doc_id'] = doc_id
            all_chunks.extend(retrieved_chunks)

    # Sort all collected chunks by score to find the best ones across all docs
    all_chunks.sort(key=lambda x: x['score'], reverse=True)

    # Use the top 5 chunks from the combined list as context
    top_chunks = all_chunks[:5]

    relevant_texts = [chunk['text'] for chunk in top_chunks if chunk['score'] > 0.5]
    answer = generate_rag_response(payload.q, relevant_texts)

    relevant_sources = [chunk for chunk in top_chunks if chunk['score'] > 0.5]

    return QueryResponse(answer=answer, sources=relevant_sources)


# --- Main Execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
