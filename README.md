# DocQA

DocQA is a question-answering tool that can answer questions about documents and URLs. It is designed to be a lightweight, standalone tool that can be easily integrated with other tools.

## Features

*   **Question Answering:** Ask questions about your documents and get answers in natural language.
*   **URL Ingestion:** Ingest documents from URLs.
*   **File Upload:** Upload your own documents.
*   **MCP-Native:** DocQA is an MCP-native tool, which means it can be easily integrated with other MCP-compatible tools.

## How to Use

### As a Standalone QA App

1.  **Install the dependencies:**

```bash
pip install -r requirements.txt
```

2.  **Run the application:**

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

3.  **Use the `/ingest` and `/query` endpoints to interact with the tool.**

    *   **`/ingest`:** Ingest a document from a URL or by uploading a file.
    *   **`/query`:** Ask a question about the ingested documents.

### As an MCP Server

1.  **Install the dependencies:**

```bash
pip install -r requirements.txt
```

2.  **Run the application:**

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

3.  **Send requests to the `/mcp` endpoint.**

## MCP Implementation

DocQA implements the following MCP actions:

*   **`get_capabilities`:** Returns a list of the tool's capabilities.
*   **`query`:** Answers questions about documents.
*   **`ingest`:** Ingests a document from a URL or content.

### `get_capabilities`

**Request:**

```json
{
    "context": {
        "request_id": "123"
    },
    "request": [
        {
            "request_id": "456",
            "action": "get_capabilities"
        }
    ]
}
```

**Response:**

```json
{
    "request_id": "123",
    "response": [
        {
            "request_id": "456",
            "response": {
                "status": "success",
                "data": {
                    "capabilities": [
                        {
                            "action": "query",
                            "description": "Answer questions about documents",
                            "input": {
                                "q": "string"
                            },
                            "output": {
                                "answer": "string",
                                "sources": "list[string]",
                                "chain_of_thought": "list[string]"
                            }
                        },
                        {
                            "action": "ingest",
                            "description": "Ingest a document from a URL or content",
                            "input": {
                                "url": "string",
                                "content": "string"
                            },
                            "output": {
                                "doc_id": "string",
                                "chunks": "int"
                            }
                        }
                    ]
                }
            }
        }
    ]
}
```

### `query`

**Request:**

```json
{
    "context": {
        "request_id": "123"
    },
    "request": [
        {
            "request_id": "456",
            "action": "query",
            "body": {
                "q": "When did World War I start?"
            }
        }
    ]
}
```

**Response:**

```json
{
    "request_id": "123",
    "response": [
        {
            "request_id": "456",
            "response": {
                "status": "success",
                "data": {
                    "answer": "World War I started on July 28, 1914.",
                    "sources": [
                        "The war lasted until November 11, 1918."
                    ],
                    "chain_of_thought": [
                        "Found 1 relevant chunks.",
                        "Prompt: ...",
                        "Generated answer."
                    ]
                }
            }
        }
    ]
}
```

### `ingest`

**Request:**

```json
{
    "context": {
        "request_id": "123"
    },
    "request": [
        {
            "request_id": "456",
            "action": "ingest",
            "body": {
                "url": "https://en.wikipedia.org/wiki/World_War_I"
            }
        }
    ]
}
```

**Response:**

```json
{
    "request_id": "123",
    "response": [
        {
            "request_id": "456",
            "response": {
                "status": "success",
                "data": {
                    "doc_id": "12345678",
                    "chunks": 10
                }
            }
        }
    ]
}
```
