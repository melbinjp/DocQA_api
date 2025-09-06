---
title: DocQA
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# DocQA: A Stateless, Session-Based, Multilingual Q&A API

DocQA is a powerful, lightweight API for building advanced question-answering applications. It allows clients to create user sessions, manage collections of documents within those sessions, and perform powerful semantic searches across single or multiple documents.

## Architecture Overview

This project is a prototype-level application built with the following components:
*   **Backend:** A Python [FastAPI](https://fastapi.tiangolo.com/) server.
*   **Embeddings:** The `sentence-transformers` library is used to generate vector embeddings for document chunks.
*   **Vector Search:** [FAISS](https://faiss.ai/) from Meta AI provides efficient in-memory similarity search.
*   **LLM:** Google's [Gemini](https://deepmind.google/technologies/gemini/) family of models is used for generating answers based on retrieved context.
*   **Session Storage:** All user sessions and document data are stored **in-memory** and are not persisted. Sessions automatically expire after a period of inactivity. This makes the server stateless but not suitable for production use without modification.

## Core Features
- **User Sessions:** Create isolated sessions for each user, allowing them to work with a private collection of documents.
- **Multi-Document Q&A:** Ingest multiple documents into a single session and perform semantic searches across the entire collection.
- **Multilingual:** Thanks to a powerful cross-lingual embedding model, you can ingest documents and ask questions in many different languages.
- **Stateless with Timeouts:** The server is stateless and does not persist any data to disk. All sessions are held in memory and are automatically cleared after 15 minutes of inactivity.

## API Workflow and Frontend Guide

Building a client application follows this logical flow:

1.  **Create a User Session:** The first step for any new user is to create a session.
    -   `POST /sessions` -> returns a `session_id`.
    -   The frontend should store this `session_id` for the duration of the user's visit.

2.  **Ingest Documents:** The user can upload multiple documents into their session.
    -   `POST /sessions/{session_id}/ingest` with a file -> returns a `doc_id`.
    -   The frontend should keep a list of the documents the user has ingested, mapping the `doc_id` to its filename.

3.  **Ask Questions:** The user can now ask questions.
    -   `POST /sessions/{session_id}/query` with a question and an optional list of `doc_ids`.
    -   If no `doc_ids` are provided, the search runs across all documents in the session.
    -   If `doc_ids` are provided, the search is limited to that subset.

4.  **Manage Documents:** The user can remove documents they no longer need.
    -   `DELETE /sessions/{session_id}/documents/{doc_id}`

---

## API Reference

### `POST /sessions`
Creates a new, empty user session.
- **Response `200 OK`:**
  ```json
  {
    "session_id": "string"
  }
  ```

### `POST /sessions/{session_id}/ingest`
Ingests a new document into the specified user session.
- **Request:** `multipart/form-data` with a `file` or `application/json` with a `url`.
- **Response `200 OK`:**
  ```json
  {
    "doc_id": "string",
    "source": "string"
  }
  ```

### `POST /sessions/{session_id}/query`
Asks a question within the user session.
- **Request Body:**
  ```json
  {
    "q": "string",
    "doc_ids": ["string"], // Optional. If omitted, searches all docs in session.
    "stream": false // Optional. Set to true for a streaming response.
  }
  ```
- **Response (Standard): `200 OK`** (`application/json`)
  When `stream` is `false` or omitted, the response is a single JSON object:
  ```json
  {
    "answer": "string",
    "sources": [
      {
        "text": "string",
        "score": "float",
        "doc_id": "string",
        "source": "string"
      }
    ]
  }
  ```
- **Response (Streaming): `200 OK`** (`text/event-stream`)
  When `stream` is `true`, the response is a Server-Sent Events (SSE) stream. The client should listen for events on this stream. Each event is a JSON object.
    1.  **Sources Event:** The first event contains the source documents that will be used to generate the answer.
        ```
        data: {"type": "sources", "data": [{"text": "...", "score": ...}]}
        ```
    2.  **Token Events:** A series of events, each containing a piece of the generated answer.
        ```
        data: {"token": "The"}
        data: {"token": " answer"}
        data: {"token": " is..."}
        ```
    3.  **End Event:** The final event signals that the stream is complete.
        ```
        data: {"type": "end"}
        ```

### `DELETE /sessions/{session_id}/documents/{doc_id}`
Deletes a specific document from a user session.
- **Response `204 No Content`** on success.
- **Response `404 Not Found`** if the session or document does not exist.

### `GET /sessions/{session_id}/status`
Checks session status and remaining time before expiration.
- **Response `200 OK`:**
  ```json
  {
    "session_id": "string",
    "active": true,
    "remaining_minutes": 12.5,
    "last_accessed": "2024-01-01T12:00:00"
  }
  ```
- Returns `active: false` if session doesn't exist or has expired.
- `remaining_minutes` only present for active sessions.

### `POST /sessions/{session_id}/refresh`
Refreshes a session to extend its timeout period.
- **Response `200 OK`:**
  ```json
  {
    "session_id": "string",
    "refreshed_at": "2024-01-01T12:00:00",
    "remaining_minutes": 15.0
  }
  ```
- **Response `404 Not Found`** if the session does not exist.

### `GET /sessions/{session_id}/health`
Simple health check for session existence and activity.
- **Response `200 OK`:** `{"status": "active"}` if session is active.
- **Response `404 Not Found`** if session doesn't exist.
- **Response `410 Gone`** if session exists but has expired.

---

## Session Management for Frontend Applications

The API provides session management endpoints to help frontend applications handle session lifecycles, timeouts, and user experience.

### Basic Usage
```javascript
// Create and manage a session
const { session_id } = await fetch('/sessions', { method: 'POST' }).then(r => r.json());

// Check session status
const status = await fetch(`/sessions/${session_id}/status`).then(r => r.json());
if (status.active) {
    console.log(`${status.remaining_minutes} minutes remaining`);
}

// Refresh session to extend timeout
await fetch(`/sessions/${session_id}/refresh`, { method: 'POST' });

// Quick health check
const health = await fetch(`/sessions/${session_id}/health`);
if (health.ok) console.log('Session active');
```

### Recommended Patterns
- **Periodic checks:** Monitor session status every 5 minutes
- **Auto-refresh:** Extend session on user activity when < 5 minutes remain
- **Error handling:** Handle 404 (not found) and 410 (expired) responses appropriately
