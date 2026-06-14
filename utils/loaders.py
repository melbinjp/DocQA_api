"""Minimal document loaders for the DocQA application."""

import os
import tempfile
from markitdown import MarkItDown
from .exceptions import DocumentLoaderError

def load_source(raw: bytes, ext: str) -> str:
    """
    Extracts text content from a raw byte stream using MarkItDown.

    Args:
        raw: The raw bytes of the file.
        ext: The file extension (e.g., '.txt', 'pdf', '.html').

    Returns:
        The extracted text content as a Markdown string.

    Raises:
        DocumentLoaderError: If there's an error during parsing.
    """
    ext = ext.lower().strip('.')
    if ext == 'url':
        ext = 'html'

    md = MarkItDown()

    # MarkItDown prefers working with files, so we write to a temp file
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        result = md.convert(tmp_path)
        if not result.text_content or not result.text_content.strip():
            raise DocumentLoaderError("Extraction resulted in no content.")
        return result.text_content
    except Exception as e:
        if isinstance(e, DocumentLoaderError):
            raise
        raise DocumentLoaderError(f"Failed to load content with extension '{ext}': {e}") from e
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass