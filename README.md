# Gemini RAG Demo

A lightweight Retrieval-Augmented Generation (RAG) micro-service built with **FastAPI** and **Google Gemini Flash**.  
It ingests PDFs, DOCX, raw text or web URLs, creates MiniLM embeddings (in-memory), and answers questions in a single request.

---

## 🚀 Live on Hugging Face Spaces

This repo is ready to be pushed to [HF Spaces](https://huggingface.co/spaces). The Space is configured via `space.yaml` (see below) to run inside a tiny Docker container.

### Space configuration ( `space.yaml` )
```yaml
# already committed to the repo
title: Gemini RAG Demo
emoji: "📄"
colorFrom: indigo
colorTo: blue
sdk: docker           # we build a tiny Docker image so FastAPI can run with uvicorn
app_port: 7860        # the app listens here (see app.py)
license: apache-2.0
python_version: "3.10"
```

### Secrets
Add your **Google AI Studio** API key to the Space secrets:
```
VARIABLE         VALUE
──────────────────────────────────────────
GOOGLE_API_KEY   <your-key-here>
```

---

## ✨ Features
* **/ingest** – upload a file (PDF, DOCX, TXT) or JSON `{url: "…"}` and store up to 30 × 500-char chunks in RAM.
* **/query**  – ask any question; the service retrieves the top 3 relevant chunks with cosine similarity and lets Gemini Flash answer.
* **/health** – tiny health/readiness probe.
* **Single-page UI** ➜ open the root URL to upload, ask, and see results.
* No heavy native deps; runs comfortably in the free CPU tier (~<120 MB once built).

---

## 🖥️ Local quick-start
```bash
# clone this repo, then
cd gemini_rag
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
export GOOGLE_API_KEY=<your-key>
uvicorn app:app --host 0.0.0.0 --port 7860
```
Open http://localhost:7860/ to interact.

---

## API
| Method | Path      | Description                                                      |
|--------|-----------|------------------------------------------------------------------|
| POST   | /ingest   | `multipart/form-data` **file** _or_ JSON `{ "url": "…" }`       |
| GET    | /query    | `?q=your+question`                                               |
| GET    | /health   | Returns basic status `{status, documents, chunks}`               |

Example:
```bash
# upload a PDF
curl -F "file=@my_resume.pdf" https://your-space-url.hf.space/ingest
# ask a question
curl "https://your-space-url.hf.space/query?q=What%20skills%20do%20I%20have%3F"
```

---

## Implementation notes
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, pure Python → fast startup, no GPU).
* **Similarity**: cosine via NumPy dot product (vectors kept in RAM; fine for tiny demos).
* **LLM**: Gemini-1.5-Flash (default) but can be overridden by setting `GENAI_TEXT_MODEL` env var.
* **Limits**: 5 MB per document, 30 chunks, 500 chars each → keeps RAM stable.
* **No persistence** – restarting the Space clears the index.

---

## 📝 License
Apache 2.0 – feel free to fork and build on top. 