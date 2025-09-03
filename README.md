# DocQA: Multilingual Question-Answering API

DocQA is a lightweight, session-based, and multilingual question-answering tool. It allows a client application to upload a document or provide a URL in various languages, and then ask questions about its content in those languages.

The API is designed to be **stateless and session-isolated**. Each ingested document is handled in a separate, in-memory session that expires after 15 minutes of inactivity. This ensures user data is never shared or persisted.

## Frontend Integration Guide

Building a frontend for this API involves a simple two-step user workflow:

1.  **Document Ingestion:** The user provides a document (either by file upload or by submitting a URL). The frontend sends this to the `/ingest` endpoint.
2.  **Store the Session ID:** The API responds with a unique `doc_id`. The frontend **must** store this ID for the current user session (e.g., in component state, React Context, or browser local storage).
3.  **Question Answering:** When the user asks a question, the frontend sends the question and a list of one or more stored `doc_ids` to the unified `/query` endpoint.
4.  **Display Results:** The frontend displays the answer and sources from the `/query` response. The `sources` will be tagged with the `doc_id` and `source` filename they came from.

---

## API Reference

The base URL for the application is the root of the server (e.g., `http://localhost:7860`).

### `/ingest`

Ingests a document and prepares it for questioning.

-   **Method:** `POST`
-   **Description:** Accepts a document and creates an in-memory session.
-   **Returns:** A unique `doc_id` which acts as the session identifier.

#### Request (File Upload)
-   **Example `curl`:**
    ```bash
    curl -X POST "http://localhost:7860/ingest" -F "file=@/path/to/your/document.txt"
    ```

#### Request (URL)
-   **Example `curl`:**
    ```bash
    curl -X POST "http://localhost:7860/ingest" -H "Content-Type: application/json" -d '{"url": "https://en.wikipedia.org/wiki/DocQA"}'
    ```

#### Responses
-   **`200 OK` (Success)**
    ```json
    {
      "doc_id": "string",
      "source": "string",
      "chunks_ingested": "integer"
    }
    ```
-   **`400 Bad Request`**: If the document can't be loaded. `{"detail": "string"}`

---

### `/query` (Unified Search Endpoint)

Asks a question against one or more previously ingested documents.

-   **Method:** `POST`
-   **Description:** Takes a question (`q`) and a list of `doc_ids`. It performs a semantic search across all specified documents and generates a single answer from the combined best context. Thanks to a powerful multilingual model, the documents and the query can be in different languages.

#### Request
-   **Headers:** `Content-Type: application/json`
-   **Body Schema:**
    ```json
    {
      "doc_ids": ["string"] or "string",
      "q": "string"
    }
    ```
-   **Example `curl` (Multiple Docs):**
    ```bash
    curl -X POST "http://localhost:7860/query" -H "Content-Type: application/json" -d '{"doc_ids": ["id-1", "id-2"], "q": "Compare topics"}'
    ```
-   **Example `curl` (Single Doc):**
    ```bash
    curl -X POST "http://localhost:7860/query" -H "Content-Type: application/json" -d '{"doc_ids": "id-1", "q": "What is the topic?"}'
    ```

#### Responses
-   **`200 OK` (Success)**
    -   **Description:** The `sources` list contains chunks from any of the provided `doc_ids`, sorted by relevance. Each source is tagged with its origin `doc_id` and `source` filename.
    -   **Body Schema:**
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
-   **`422 Unprocessable Entity`**: If the request body is invalid.
