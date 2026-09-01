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
from google import genai
from google.genai import errors
import uvicorn
import httpx
import numpy as np

# Set Hugging Face Hub download timeout to 120 seconds to prevent ReadTimeoutErrors in Spaces
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from sentence_transformers import SentenceTransformer
from utils.loaders import load_source, load_source_pages
from utils.url_fetch import fetch_url_document, is_safe_url_async
from utils.prompting import build_manifest, label_chunk
from utils.vision import merge as vision_merge, pages_for_vision, transcribe_pages
from utils.splitter import split_text, split_pages, index_entries
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    app.state.http_client = httpx.AsyncClient(headers=headers)
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
# Initialize the modern google-genai Client
ai_client = genai.Client(api_key=GENAI_API_KEY)
MODEL_NAME = "gemini-3.5-flash"
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-2.5-flash"]

# Two ceilings that were inline literals, named so they are visible in one place.
# 30s is the value the code already used. Warm calls to this Space come back in a
# few seconds; the ones that run past 30s are the ones right after the Space wakes
# from sleep, and those are now retried rather than surfaced. Raising the ceiling
# instead would have made every genuinely stuck call three times slower to report.
LLM_TIMEOUT_SECONDS = 30.0

# How many chunks are retrieved per document, and how many survive the merge to
# become the prompt. These were 5 and a bare `[:5]` literal further down, which
# meant raising one silently did nothing because the other still truncated.
#
# Eight rather than five because chunks are now 1500 characters of clean prose
# rather than 500 characters cut mid-sentence, so the marginal chunk is worth
# reading. Eight of them is around 12k characters, which is a small prompt for
# the model receiving it.
#
# The merge across documents sorts by score, and that only became sound when the
# index moved to cosine: the old 1/(1+l2) score was magnitude-biased, so scores
# from two documents with different chunk lengths were not on the same scale.
# Looking at a page costs a model call, so it is off by default for anyone
# running this without the quota for it, and on here because the alternative is
# rejecting every scanned document outright.
VISION_ENABLED = os.getenv("VISION_ENABLED", "1") not in ("0", "false", "False")

RETRIEVE_PER_DOC = 8
CONTEXT_CHUNKS = 8

# `wait_for` around `generate_content_stream(...)` bounds getting the iterator, not
# consuming it. This bounds each chunk, so a stream that stalls mid-answer ends
# instead of leaving the browser on an open connection with no error and no end.
STREAM_CHUNK_TIMEOUT = 30.0

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
        "Answer the following question using only the material below. It begins "
        "with the list of documents loaded in this session, then excerpts, each "
        "headed by the document and page it came from. When the question is "
        "about which document says or covers something, answer from the list of "
        "documents, not from excerpts: a reference list inside a paper names "
        "other works and is not itself a document in this session. "
        "If the excerpts do not answer the question, say so instead of guessing. "
        "If the user asks in a language other than English, respond in their language.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    max_retries = 3
    retry_delay = 1.0
    emitted = False

    for attempt in range(max_retries):
        current_model = MODEL_NAME
        if attempt > 0 and attempt - 1 < len(FALLBACK_MODELS):
            current_model = FALLBACK_MODELS[attempt - 1]

        last_attempt = attempt == max_retries - 1

        try:
            if stream:
                response = await asyncio.wait_for(
                    ai_client.aio.models.generate_content_stream(
                        model=current_model,
                        contents=prompt
                    ),
                    timeout=LLM_TIMEOUT_SECONDS
                )
                iterator = response.__aiter__()
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            iterator.__anext__(), timeout=STREAM_CHUNK_TIMEOUT
                        )
                    except StopAsyncIteration:
                        break
                    # Ensure the chunk has content before sending
                    if chunk.text:
                        emitted = True
                        yield f"data: {json.dumps({'token': chunk.text})}\n\n"
                return  # Exit generator on success
            else:
                response = await asyncio.wait_for(
                    ai_client.aio.models.generate_content(
                        model=current_model,
                        contents=prompt
                    ),
                    timeout=LLM_TIMEOUT_SECONDS
                )
                yield response.text.strip()
                return  # Exit generator on success
        except (asyncio.TimeoutError, errors.APIError) as e:
            # A timeout and a 429/503 are the same kind of problem: the answer was
            # never produced, so asking again is safe. Anything else is a real
            # error, and retrying it three times only delays telling the user.
            if isinstance(e, errors.APIError) and e.code not in (429, 503):
                error_message = f"LLM generation failed: {e.message}"
                if stream:
                    yield f"data: {json.dumps({'error': error_message})}\n\n"
                    return
                else:
                    raise HTTPException(status_code=500, detail=error_message)

            timed_out = isinstance(e, asyncio.TimeoutError)

            # Once tokens are on the wire the answer is half delivered, and
            # starting over would repeat what the reader has already seen. A
            # stream that fails mid-answer ends with an error, not a second try.
            if not last_attempt and not emitted:
                reason = "Timed out" if timed_out else "High demand hit"
                print(f"{reason} for {current_model}. Retrying with fallback... "
                      f"(Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
                continue

            if timed_out:
                error_message = "LLM generation timed out."
                status_code = 504
            else:
                error_message = "Model is experiencing high demand. Please try again later."
                status_code = 503
            if stream:
                yield f"data: {json.dumps({'error': error_message})}\n\n"
                return
            else:
                raise HTTPException(status_code=status_code, detail=error_message)
        except Exception as e:
            error_message = f"LLM generation failed: {e}"
            if stream:
                yield f"data: {json.dumps({'error': error_message})}\n\n"
                return
            else:
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
    page: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[QuerySource]

class SessionStatusResponse(BaseModel):
    session_id: str
    active: bool
    remaining_minutes: Optional[float] = None
    last_accessed: str

class SessionRefreshResponse(BaseModel):
    session_id: str
    refreshed_at: str
    remaining_minutes: float

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
async def ingest(session_id: str, request: Request):
    with _session_lock:
        user_session = sessions.get(session_id)
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")

    file_filename = None
    file_content = None
    url = None
    has_file = False

    body_bytes = await request.body()

    # 1. Try to parse as JSON URL first
    try:
        body = json.loads(body_bytes)
        url = body.get("url")
    except Exception:
        pass

    # 2. If no URL was successfully parsed, try parsing as Form data
    if not url:
        try:
            form = await request.form()
            file = form.get("file")
            if file and hasattr(file, "filename") and file.filename:
                file_filename = file.filename
                file_content = await file.read()
                has_file = True
        except Exception:
            pass

    if not has_file and not url:
        content_type = request.headers.get("content-type", "")
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provide either a file (multipart/form-data) or a URL (application/json). "
                f"Received Content-Type: {content_type}."
            )
        )
    if has_file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL, not both.")

    source_name = ""
    content = b""
    source_ext = "url"

    if has_file:
        source_name = file_filename
        content = file_content
        source_ext = pathlib.Path(source_name).suffix or "url"
    elif url:
        if not await is_safe_url_async(url):
            raise HTTPException(status_code=400, detail="Invalid or restricted URL provided.")
        source_name = url
        content, source_ext = await fetch_url_document(app.state.http_client, url)

    try:
        pages = load_source_pages(content, source_ext)
    except DocumentLoaderError as e:
        # Hold the failure rather than raising it. For a PDF this is usually a
        # scan, which has no text layer to extract and which the vision pass
        # below can read. If that finds nothing either, the error is raised then.
        if source_ext.lower().strip(".") != "pdf":
            raise HTTPException(status_code=400, detail=str(e))
        pages = []

    # Look at the pages a text extractor cannot read: scans, charts, and tables
    # it mangles. Additive, and never allowed to fail the ingest on its own.
    if source_ext.lower().strip(".") == "pdf" and VISION_ENABLED:
        try:
            targets = await asyncio.to_thread(pages_for_vision, content)
            if targets:
                transcripts = await transcribe_pages(ai_client, MODEL_NAME, targets)
                if transcripts:
                    pages = vision_merge(pages, transcripts)
                    print(f"Vision read {len(transcripts)} of {len(targets)} pages.")
        except Exception as e:
            print(f"Vision pass skipped: {e}")

    if not pages or not any(t and t.strip() for _, t in pages):
        raise HTTPException(status_code=400, detail="Could not extract any text from the provided source.")

    page_chunks = split_pages(pages)
    chunks = [c["text"] for c in page_chunks]
    chunk_pages = [c["page"] for c in page_chunks]

    # What is matched is not always what is read. A table chunk carries an
    # `embed_text` of just its caption, because a vector built from the whole
    # grid is dominated by the numbers and stops being findable: measured
    # 2026-09-01, the Table 3 chunk was not retrieved for a question its own
    # caption answers, while its neighbours scored 0.36 to 0.47. The full grid
    # is still what goes into the prompt and still what gets cited.
    # One chunk can own several vectors. A long chunk is indexed whole and again
    # window by window, so a fact buried inside a chunk about something else is
    # still reachable; whichever vector matches, the parent chunk is returned.
    entries = index_entries(page_chunks)
    embed_inputs = [text for text, _ in entries]
    vector_parents = [parent for _, parent in entries]
    if not chunks:
        raise HTTPException(status_code=400, detail="The document is too short to be processed.")

    # --- Caching and Embedding Logic ---
    # This logic checks the user's session cache for existing chunk embeddings.
    # It only sends chunks that have not been seen before to the embedding model,
    # avoiding redundant, expensive computations.
    ordered_embeddings = [None] * len(embed_inputs)
    chunks_to_encode = []
    indices_of_new_chunks = []

    # Identify which chunks are new and which are cached. Keyed on what is
    # actually encoded, not on the chunk text, or a table would be looked up by
    # its grid and stored under its caption.
    for i, chunk in enumerate(embed_inputs):
        if chunk in user_session.embedding_cache:
            ordered_embeddings[i] = user_session.embedding_cache[chunk]
        else:
            chunks_to_encode.append(chunk)
            indices_of_new_chunks.append(i)

    # If there are new chunks, encode them in a single batch for efficiency
    if chunks_to_encode:
        # Encode each unique new chunk only once to save computation
        unique_new_chunks = list(dict.fromkeys(chunks_to_encode))

        try:
            generated_embeddings = await asyncio.wait_for(
                asyncio.to_thread(
                    app.state.embedding_model.encode, unique_new_chunks, convert_to_numpy=True
                ),
                timeout=180.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Embedding generation timed out.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

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
    rag_session.ingest(chunks, all_embeddings_np, chunk_pages, vector_parents)
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
        try:
            retrieved = await rag_session.query(payload.q, k=RETRIEVE_PER_DOC)
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query search failed: {e}")

        for chunk in retrieved:
            chunk['doc_id'] = doc_id
            chunk['source'] = rag_session.source
        all_chunks.extend(retrieved)

    all_chunks.sort(key=lambda x: x['score'], reverse=True)
    top_chunks = all_chunks[:CONTEXT_CHUNKS]

    relevant_sources = [QuerySource(**chunk) for chunk in top_chunks]
    # Label every excerpt with where it came from.
    #
    # The prompt used to be the chunk texts joined by blank lines and nothing
    # else, so the model was never told which document it was reading. With one
    # document that is merely wasteful. With eight in a session it makes a whole
    # class of question unanswerable, and measured 2026-09-01 it did: asked
    # which of the loaded papers was about image recognition, the answer listed
    # three entries out of ResNet's own bibliography instead of naming ResNet,
    # because a chunk of reference list and a chunk of a paper look identical
    # when neither says what it is. It also risks quietly attributing one
    # document's numbers to another.
    relevant_texts = [label_chunk(chunk) for chunk in top_chunks]

    # And always say what is loaded, whatever retrieval happened to return. A
    # question about the set of documents cannot be answered from chunks: the
    # chunks that match "image recognition" best are a reference list, not the
    # abstract of the paper that is about it.
    manifest = build_manifest(
        (rag.source, rag.chunks[0] if rag.chunks else "")
        for rag in user_session.docs.values()
    )
    if manifest:
        relevant_texts = [manifest] + relevant_texts

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

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # If not streaming, use the original logic
    else:
        answer = ""
        # The async generator yields one result in non-streaming mode
        async for content in generate_rag_response(payload.q, relevant_texts, stream=False):
            answer = content
        return QueryResponse(answer=answer, sources=relevant_sources)

@app.get("/sessions/{session_id}/status", response_model=SessionStatusResponse, summary="Get session status and remaining time")
async def get_session_status(session_id: str):
    """Returns session status, activity state, and remaining time before expiration."""
    now = datetime.datetime.now()
    expiration_time = datetime.timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    
    with _session_lock:
        user_session = sessions.get(session_id)
    
    if not user_session:
        return SessionStatusResponse(
            session_id=session_id,
            active=False,
            last_accessed=now.isoformat()
        )
    
    time_since_access = now - user_session.last_accessed
    remaining_time = expiration_time - time_since_access
    
    if remaining_time.total_seconds() <= 0:
        return SessionStatusResponse(
            session_id=session_id,
            active=False,
            last_accessed=user_session.last_accessed.isoformat()
        )
    
    return SessionStatusResponse(
        session_id=session_id,
        active=True,
        remaining_minutes=remaining_time.total_seconds() / 60,
        last_accessed=user_session.last_accessed.isoformat()
    )

@app.post("/sessions/{session_id}/refresh", response_model=SessionRefreshResponse, summary="Refresh session to extend timeout")
async def refresh_session(session_id: str):
    """Refreshes a session to extend its timeout period."""
    with _session_lock:
        user_session = sessions.get(session_id)
    
    if not user_session:
        raise HTTPException(status_code=404, detail="User session not found.")
    
    user_session.touch()
    
    return SessionRefreshResponse(
        session_id=session_id,
        refreshed_at=user_session.last_accessed.isoformat(),
        remaining_minutes=SESSION_TIMEOUT_MINUTES
    )

@app.get("/sessions/{session_id}/health", summary="Simple session health check")
async def session_health_check(session_id: str):
    """Simple endpoint to check if session exists and is active."""
    now = datetime.datetime.now()
    expiration_time = datetime.timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    
    with _session_lock:
        user_session = sessions.get(session_id)
    
    if not user_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    time_since_access = now - user_session.last_accessed
    if time_since_access > expiration_time:
        raise HTTPException(status_code=410, detail="Session expired")
    
    return {"status": "active"}

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
