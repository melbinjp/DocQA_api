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
        text_content = ""
        try:
            result = md.convert(tmp_path)
            text_content = result.text_content or ""
        except Exception as e:
            if ext == "pdf":
                # Suppress error and let PyMuPDF handle fallback below
                pass
            else:
                raise DocumentLoaderError(f"Failed to load content with extension '{ext}': {e}") from e

        # If PDF extraction returned no content or failed, run PyMuPDF fallback
        if ext == "pdf" and (not text_content or not text_content.strip()):
            try:
                import fitz
                doc = fitz.open(tmp_path)
                texts = [page.get_text() for page in doc]
                text_content = "\n".join(texts)
            except Exception as pdf_err:
                raise DocumentLoaderError(
                    f"PDF extraction failed using both MarkItDown and PyMuPDF fallback. "
                    f"PyMuPDF error: {pdf_err}"
                )

        if not text_content or not text_content.strip():
            raise DocumentLoaderError("Extraction resulted in no content.")
        return text_content
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass