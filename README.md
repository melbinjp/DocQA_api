---
title: DocQA
emoji: 🤖
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---


# DocQA: Gemini RAG Microservice

**DocQA** is a lightweight Retrieval-Augmented Generation (RAG) microservice built with FastAPI and Google Gemini Flash. It ingests PDFs, DOCX, raw text, or web URLs, creates MiniLM embeddings (in-memory), and answers questions in a single request.

---

## 🚀 Live Demo

- **API on Hugging Face Spaces:** [melbinjp/DocQA](https://huggingface.co/spaces/melbinjp/DocQA)
- **Web UI:** [docqa.wecanuseai.com](https://docqa.wecanuseai.com) ([GitHub Pages](https://melbinjp.github.io/DocQA/))

---

## Features

- Upload and ingest PDF, DOCX, TXT, or web URLs
- In-memory semantic search with MiniLM embeddings
- FastAPI backend with OpenAPI docs
- Google Gemini Flash for answer generation
- Simple REST API for integration

---

## Usage

### 1. Ingest a Document or URL

**POST** `/ingest`

**Form-data:**
- `file`: Upload a document (PDF, DOCX, TXT)

**or JSON:**
```json
{
  "url": "https://example.com/article"
}
```

**Response:**
```json
{
  "doc_id": "abcd1234",
  "chunks": 18
}
```

### 2. Query

**GET** `/query?q=your+question`

**Response:**
```json
{
  "answer": "...",
  "sources": ["chunk1", "chunk2"]
}
```

---

## Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Or build/run with Docker (used by Hugging Face Spaces):
```bash
docker build -t docqa .
docker run -p 7860:7860 docqa
```

---

## Deployment

This repo is ready to be pushed to [this HF Space](https://huggingface.co/spaces/melbinjp/DocQA). The Space is configured to run inside a Docker container.

### Required Secret
Add your **Google AI Studio** API key to the Space secrets: