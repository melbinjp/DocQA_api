"""Turning retrieved chunks into the text the model is asked to read.

Kept out of `app.py` so it can be tested without importing the embedding model.
"""


def short_source(source: str) -> str:
    """A source a reader would recognise: a filename, not a URL with a query."""
    if not source:
        return "document"
    name = source.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    return name or source


def label_chunk(chunk: dict) -> str:
    """One retrieved excerpt, headed by the document and page it came from."""
    name = short_source(chunk.get("source") or "")
    page = chunk.get("page")
    where = f"{name}, page {page}" if page is not None else name
    return f"[From {where}]\n{chunk['text']}"
