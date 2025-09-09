"""QA RAG application with multi-document user sessions."""
import os
import uuid
import pathlib
import datetime
import asyncio
import threading
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union

import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
import uvicorn
import httpx
import numpy as np

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
_session_lock = threading.Lock()

# --- Background Cleanup Logic ---
def _clean_sessions_once():
    now = datetime.datetime.now()
    expiration_time = datetime.timedelta(minutes=SESSION_TIMEOUT_MINUTES)

    with _session_lock:
        # Create a copy of the session IDs to avoid modifying the dictionary while iterating
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

    print("Initializing HTTP client...")
    app.state.http_client = httpx.AsyncClient()
    print("HTTP client initialized.")

    print("Starting session cleanup task...")
    asyncio.create_task(cleanup_expired_sessions_task())
    yield

    print("Closing HTTP client...")
    await app.state.http_client.aclose()
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
async def generate_rag_response(query: str, context_chunks: List[str], stream: bool = False):
    """Generates a response from the LLM, supports streaming."""
    if not context_chunks:
        if stream:
            yield f"data: {json.dumps({'token': 'No relevant information found.'})}\n\n"
        else:
            yield "No relevant information found."
        return

    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer the following question based only on the provided context. "
        "If the user asks in a language other than English, respond in their language.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    try:
        response = llm_model.generate_content(prompt, stream=stream)
        if stream:
            for chunk in response:
                # Ensure the chunk has content before sending
                if chunk.text:
                    yield f"data: {json.dumps({'token': chunk.text})}\n\n"
        else:
            yield response.text.strip()
    except Exception as e:
        error_message = f"LLM generation failed: {e}"
        if stream:
            yield f"data: {json.dumps({'error': error_message})}\n\n"
        else:
            # In a non-streaming context, we can raise an exception that the
            # calling endpoint can handle.
            raise HTTPException(status_code=500, detail=error_message)

# --- API Models ---
class SessionResponse(BaseModel):
    session_id: str

class IngestResponse(BaseModel):
    doc_id: str
    source: str
    num_chunks: int

class QueryPayload(BaseModel):
    q: str
    doc_ids: Optional[List[str]] = None
    stream: Optional[bool] = False

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
    with _session_lock:
        sessions[session_id] = UserSession()
    return SessionResponse(session_id=session_id)

@app.post("/sessions/{session_id}/ingest", response_model=IngestResponse, summary="Ingest a document into a session")
async def ingest(session_id: str, request: Request, file: Optional[UploadFile] = File(None)):
    with _session_lock:
        user_session = sessions.get(session_id)
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")

    url = None
    # If a file is not provided, the client may have sent a JSON body with a URL.
    if not file:
        try:
            body = await request.json()
            url = body.get("url")
        except Exception:
            # This can happen if the body is not valid JSON, which is expected
            # if the client sent an empty multipart form.
            pass

    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide either a file (multipart/form-data) or a URL (application/json).")
    if file and url:
        # This case should ideally not be hit if the client is behaving.
        raise HTTPException(status_code=400, detail="Provide either a file or a URL, not both.")

    source_name = ""
    content = b""

    if file:
        source_name = file.filename
        content = await file.read()
    elif url:
        source_name = url
        try:
            response = await app.state.http_client.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            content = response.content
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Failed to fetch URL: {e.response.text}")
        except httpx.RequestError as e:
            # For other request errors like connection issues
            raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {e}")

    try:
        text = load_source(content, pathlib.Path(source_name).suffix or "url")
    except DocumentLoaderError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the provided source.")

    chunks = split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="The document is too short to be processed.")

    # --- Caching and Embedding Logic ---
    # This logic checks the user's session cache for existing chunk embeddings.
    # It only sends chunks that have not been seen before to the embedding model,
    # avoiding redundant, expensive computations.
    ordered_embeddings = [None] * len(chunks)
    chunks_to_encode = []
    indices_of_new_chunks = []

    # Identify which chunks are new and which are cached
    for i, chunk in enumerate(chunks):
        if chunk in user_session.embedding_cache:
            ordered_embeddings[i] = user_session.embedding_cache[chunk]
        else:
            chunks_to_encode.append(chunk)
            indices_of_new_chunks.append(i)

    # If there are new chunks, encode them in a single batch for efficiency
    if chunks_to_encode:
        # Encode each unique new chunk only once to save computation
        unique_new_chunks = list(dict.fromkeys(chunks_to_encode))

        generated_embeddings = app.state.embedding_model.encode(
            unique_new_chunks, convert_to_numpy=True
        )

        new_embeddings_dict = {
            chunk: emb for chunk, emb in zip(unique_new_chunks, generated_embeddings)
        }

        # Add the newly generated embeddings to the session cache for future use
        user_session.embedding_cache.update(new_embeddings_dict)

        # Place the new embeddings into the final ordered list
        for i, chunk in enumerate(chunks_to_encode):
            original_index = indices_of_new_chunks[i]
            ordered_embeddings[original_index] = new_embeddings_dict[chunk]

    all_embeddings_np = np.array(ordered_embeddings)
    # --- End of Caching and Embedding Logic ---

    doc_id = uuid.uuid4().hex
    rag_session = RAGSession(source=source_name, embedding_model=app.state.embedding_model)
    # Pass the pre-computed embeddings to the new ingest method
    rag_session.ingest(chunks, all_embeddings_np)
    user_session.add_doc(doc_id, rag_session)

    return IngestResponse(doc_id=doc_id, source=source_name, num_chunks=len(chunks))

@app.post("/sessions/{session_id}/query", summary="Ask a question within a session")
async def query(session_id: str, payload: QueryPayload):
    with _session_lock:
        user_session = sessions.get(session_id)
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")

    user_session.touch()

    # Determine which documents to query.
    docs_to_query_items = user_session.docs.items()
    if payload.doc_ids:
        docs_to_query_items = [
            (doc_id, user_session.get_doc(doc_id))
            for doc_id in payload.doc_ids
            if user_session.get_doc(doc_id) is not None
        ]

    all_chunks = []
    for doc_id, rag_session in docs_to_query_items:
        retrieved = rag_session.query(payload.q, k=5)
        for chunk in retrieved:
            chunk['doc_id'] = doc_id
            chunk['source'] = rag_session.source
        all_chunks.extend(retrieved)

    all_chunks.sort(key=lambda x: x['score'], reverse=True)
    top_chunks = all_chunks[:5]

    relevant_sources = [QuerySource(**chunk) for chunk in top_chunks]
    relevant_texts = [chunk['text'] for chunk in top_chunks]

    # If streaming is requested, return a StreamingResponse
    if payload.stream:
        async def stream_generator():
            # First, send an event with the sources
            sources_data = [s.model_dump() for s in relevant_sources]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

            # Then, stream the LLM response tokens
            async for chunk in generate_rag_response(payload.q, relevant_texts, stream=True):
                yield chunk

            # Signal the end of the stream
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # If not streaming, use the original logic
    else:
        answer = ""
        # The async generator yields one result in non-streaming mode
        async for content in generate_rag_response(payload.q, relevant_texts, stream=False):
            answer = content
        return QueryResponse(answer=answer, sources=relevant_sources)

@app.delete("/sessions/{session_id}/documents/{doc_id}", status_code=204, summary="Delete a document from a session")
async def delete_document(session_id: str, doc_id: str):
    """Deletes a specific document from a user session."""
    with _session_lock:
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
