# DocQA: A Stateless, Session-Based, Multilingual Q&A API

DocQA is a powerful, lightweight API for building advanced question-answering applications. It allows clients to create user sessions, manage collections of documents within those sessions, and perform powerful semantic searches across single or multiple documents.

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
    "doc_ids": ["string"] // Optional. If omitted, searches all docs in session.
  }
  ```
- **Response `200 OK`:**
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

### `DELETE /sessions/{session_id}/documents/{doc_id}`
Deletes a specific document from a user session.
- **Response `204 No Content`** on success.
- **Response `404 Not Found`** if the session or document does not exist.
