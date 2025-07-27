"""QA RAG in Hugging Face Spaces"""
import os
import uuid
import json
import pathlib
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import uvicorn

# Import utilities
from utils.loaders import load_source
from utils.splitter import split_text

# Configure Gemini
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GENAI_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in HF Space secrets")

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

# Small embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Simple in-memory storage
vectors = []
texts = []
doc_ids = []
manifest = {}

# Create FastAPI app
app = FastAPI(title="DocQA")

# Allow the static frontend (hosted on GitHub Pages or your custom domain) to
# call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://melbinjp.github.io",  # GitHub Pages domain
        "https://wecanuseai.com"      # Custom domain where the iframe lives

    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to UI
UI_PATH = pathlib.Path(__file__).parent / "ui.html"

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("https://melbinjp.github.io/docqa/")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the UI"""
    if UI_PATH.exists():
        return FileResponse(UI_PATH)
    return "<h1>Chat with a Doc</h1><p>Upload documents OR ADD URLsand ask questions!</p>"

class URLPayload(BaseModel):
    url: str

@app.post("/ingest")
async def ingest(request: Request, file: Optional[UploadFile] = File(None), payload: Optional[URLPayload] = Body(None)):
    """Upload a document or URL"""
    # FastAPI expects multipart when an UploadFile parameter is present. If the client
    # sent raw JSON (Content-Type: application/json) the automatic parsing above
    # leaves `payload` as None. To support both, attempt to parse JSON manually
    # before rejecting the request.
    if not file and not payload:
        # Only try to parse JSON if the content type contains "application/json"
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = await request.json()
                if isinstance(body, dict) and "url" in body:
                    payload = URLPayload(url=body["url"])
            except Exception:
                pass

    if not file and not payload:
        raise HTTPException(400, "Provide a file or URL")
    
    doc_id = uuid.uuid4().hex[:8]
    
    # Load content
    if file:
        content = await file.read()
        if len(content) > 5_000_000:  # 5MB limit
            raise HTTPException(413, "File too large (max 5MB)")
        text = load_source(content, pathlib.Path(file.filename).suffix)
        source_name = file.filename
    else:
        import requests
        # Fetch the URL with a browser-like User-Agent so sites like Wikipedia don't block us
        try:
            resp = requests.get(payload.url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (GeminiRAG)"})
        except Exception as e:
            raise HTTPException(400, f"Error fetching URL: {str(e)}")

        if resp.status_code != 200:
            raise HTTPException(400, f"Could not fetch URL (status {resp.status_code})")

        if len(resp.content) > 5_000_000:
            raise HTTPException(413, "Content too large")
        text = load_source(resp.content, "url")
        source_name = payload.url
    
    if not text:
        raise HTTPException(400, "Could not extract text")
    
    # Process text
    chunks = split_text(text, max_chars=500)[:30]  # Limit chunks
    
    # Store embeddings
    for i, chunk in enumerate(chunks):
        embedding = embed_model.encode(chunk)
        vectors.append(embedding)
        texts.append(chunk)
        doc_ids.append(f"{doc_id}_{i}")
    
    manifest[doc_id] = {"name": source_name, "chunks": len(chunks)}
    
    return {"doc_id": doc_id, "chunks": len(chunks)}

@app.get("/query")
async def query(q: str):
    """Answer questions about documents"""
    if not vectors:
        return {"answer": "No documents uploaded yet.", "sources": []}
    
    # Embed query
    query_vec = embed_model.encode(q)
    
    # Compute similarities
    similarities = []
    for vec in vectors:
        sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
        similarities.append(sim)
    
    # Get top 3
    top_indices = np.argsort(similarities)[-3:][::-1]
    
    # Build context
    # Slightly lower the similarity threshold so we don't miss relevant chunks
    context_texts = [texts[i] for i in top_indices if similarities[i] > 0.15]
    
    if not context_texts:
        return {"answer": "No relevant information found.", "sources": []}
    
    # Generate answer
    context = "\n\n".join(context_texts)
    prompt = f"""Answer this question based on the context provided.

Context:
{context}

Question: {q}

Answer:"""
    
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error generating response: {str(e)}"
    
    return {
        "answer": answer,
        "sources": context_texts[:2]
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "documents": len(manifest),
        "chunks": len(texts)
    }

# This is the key part - run the server when the script is executed
if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)