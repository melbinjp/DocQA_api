# DocQA: Stateless Question-Answering API

DocQA is a lightweight, session-based question-answering tool. It allows a client application to upload a document or provide a URL, and then ask questions about its content.

The API is designed to be **stateless and session-isolated**. Each ingested document is handled in a separate, in-memory session, ensuring user data is never shared or persisted.

## Frontend Integration Guide

Building a frontend for this API involves a simple two-step user workflow:

1.  **Document Ingestion:** The user provides a document (either by file upload or by submitting a URL). The frontend sends this to the `/ingest` endpoint.
2.  **Store the Session ID:** The API responds with a unique `doc_id`. The frontend **must** store this ID for the current user session (e.g., in component state, React Context, or browser local storage).
3.  **Question Answering:** When the user asks a question, the frontend sends the question **and** the stored `doc_id` to the `/query` endpoint.
4.  **Display Results:** The frontend displays the answer and sources from the `/query` response. If the user ingests a new document, the workflow repeats, and the old `doc_id` is replaced with the new one.

---

## API Reference

The base URL for the application is the root of the server (e.g., `http://localhost:7860`).

### `/ingest`

Ingests a document and prepares it for questioning.

-   **Method:** `POST`
-   **Description:** Accepts a document via file upload (`multipart/form-data`) or a URL (`application/json`). It processes the document into text chunks, embeds them, and creates an in-memory search index.
-   **Returns:** A unique `doc_id` which acts as the session identifier for subsequent queries.

#### Request (File Upload)
-   **Headers:** `Content-Type: multipart/form-data`
-   **Body:** A form field named `file` containing the document.
-   **Example `curl`:**
    ```bash
    curl -X POST "http://localhost:7860/ingest" \
         -F "file=@/path/to/your/document.txt"
    ```

#### Request (URL)
-   **Headers:** `Content-Type: application/json`
-   **Body Schema:**
    ```json
    {
      "url": "string"
    }
    ```
-   **Example `curl`:**
    ```bash
    curl -X POST "http://localhost:7860/ingest" \
         -H "Content-Type: application/json" \
         -d '{"url": "https://en.wikipedia.org/wiki/World_War_I"}'
    ```

#### Responses
-   **`200 OK` (Success)**
    -   **Body Schema:**
        ```json
        {
          "doc_id": "string",
          "source": "string",
          "chunks_ingested": "integer"
        }
        ```
-   **`400 Bad Request`**
    -   **Description:** Occurs if no file or URL is provided, if the URL is unreachable, or if the document content cannot be parsed.
    -   **Body Schema:** `{"detail": "string"}`
-   **`422 Unprocessable Entity`**
    -   **Description:** Occurs if the request body is not valid JSON when `Content-Type` is `application/json`.
    -   **Body Schema:** Standard FastAPI validation error response.

---

### `/query`

Asks a question against a previously ingested document.

-   **Method:** `POST`
-   **Description:** Takes a `doc_id` and a question string (`q`). It retrieves the correct document session, performs a similarity search to find relevant context, and generates an answer using an LLM.

#### Request
-   **Headers:** `Content-Type: application/json`
-   **Body Schema:**
    ```json
    {
      "doc_id": "string",
      "q": "string"
    }
    ```
-   **Example `curl`:**
    ```bash
    curl -X POST "http://localhost:7860/query" \
         -H "Content-Type: application/json" \
         -d '{"doc_id": "your-stored-doc-id", "q": "What is the main topic?"}'
    ```

#### Responses
-   **`200 OK` (Success)**
    -   **Body Schema:**
        ```json
        {
          "answer": "string",
          "sources": [
            {
              "text": "string",
              "score": "float"
            }
          ]
        }
        ```
-   **`404 Not Found`**
    -   **Description:** Occurs if the provided `doc_id` does not correspond to an active session.
    -   **Body Schema:** `{"detail": "Document session with doc_id '...' not found."}`
-   **`422 Unprocessable Entity`**
    -   **Description:** Occurs if `doc_id` or `q` are missing from the request body.
    -   **Body Schema:** Standard FastAPI validation error response.

---

### `/query-multiple`

Asks a question against a list of previously ingested documents.

-   **Method:** `POST`
-   **Description:** Takes a list of `doc_ids` and a question string (`q`). It retrieves all corresponding document sessions, performs a similarity search across all of them, and generates a single answer from the combined best context.

#### Request
-   **Headers:** `Content-Type: application/json`
-   **Body Schema:**
    ```json
    {
      "doc_ids": ["string"],
      "q": "string"
    }
    ```
-   **Example `curl`:**
    ```bash
    curl -X POST "http://localhost:7860/query-multiple" \
         -H "Content-Type: application/json" \
         -d '{"doc_ids": ["doc-id-1", "doc-id-2"], "q": "Compare the main topics"}'
    ```

#### Responses
-   **`200 OK` (Success)**
    -   **Description:** The response format is identical to the single `/query` endpoint, but the `sources` may come from any of the provided `doc_ids`.
    -   **Body Schema:**
        ```json
        {
          "answer": "string",
          "sources": [
            {
              "text": "string",
              "score": "float",
              "doc_id": "string"
            }
          ]
        }
        ```
