"""Minimal document loaders for the DocQA application."""

import os
import tempfile
from markitdown import MarkItDown
from .exceptions import DocumentLoaderError

def load_source_pages(raw: bytes, ext: str) -> list[tuple[int | None, str]]:
    """Extract text while keeping the page it came from.

    Returns `[(page_number, text), ...]`, one-based, with `None` for formats that
    have no pages. `load_source` flattens this and is kept for callers that do not
    need the page.

    **PyMuPDF is tried first for PDFs, not second.** It used to be a fallback that
    only ran when MarkItDown returned nothing at all, so a PDF MarkItDown read
    badly rather than not at all never reached it.

    Observed on the deployed Space, 2026-08-31: querying arXiv 2005.11401 returned
    source text with the spaces removed, `nenthelps toguidethegeneration`. The same
    PDF, the same pinned `markitdown==0.1.6`, extracts cleanly on a developer
    machine. The difference is that `markitdown[pdf]` pulls **pdfminer.six, which
    this project does not pin**, so the image resolves its own version at build
    time and the same document extracts differently in development and production.

    That is the real defect, and it is worse than a bad extractor: text quality
    varies by deployment and nothing downstream can tell. PyMuPDF is a pinned
    direct dependency and is deterministic here, so it goes first. Extraction that
    silently mangles is worse than extraction that fails, because a mangled chunk
    still embeds, still retrieves, and still gets quoted back to a user.
    """
    ext = ext.lower().strip('.')

    if ext == 'pdf':
        try:
            import fitz
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                doc = fitz.open(tmp_path)
                pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
                doc.close()
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            if any(text and text.strip() for _, text in pages):
                return [(n, t) for n, t in pages if t and t.strip()]
        except Exception:
            # Fall through to the generic path below rather than failing here.
            pass

    text = load_source(raw, ext)
    return [(None, text)]


def load_source(raw: bytes, ext: str) -> str:
    """
    Extracts text content from a raw byte stream using MarkItDown (for office documents)
    or BeautifulSoup (for HTML/URLs).

    Args:
        raw: The raw bytes of the file.
        ext: The file extension (e.g., '.txt', 'pdf', '.html').

    Returns:
        The extracted text content.

    Raises:
        DocumentLoaderError: If there's an error during parsing.
    """
    ext = ext.lower().strip('.')
    
    # Fast-path for plain text and markdown files
    if ext in ['md', 'txt', 'text']:
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('utf-8', errors='ignore')

    # Fast-path for HTML/URLs using BeautifulSoup to prevent timeouts on large webpages
    if ext in ['url', 'html', 'htm']:
        try:
            from bs4 import BeautifulSoup
            html = raw.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')
            # Decompose tags that do not contain main reading content
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
                tag.decompose()
            text_content = soup.get_text(' ', strip=True)
            if not text_content or not text_content.strip():
                raise DocumentLoaderError("HTML text extraction resulted in no content.")
            return text_content
        except Exception as e:
            if isinstance(e, DocumentLoaderError):
                raise
            raise DocumentLoaderError(f"Failed to parse HTML content: {e}") from e

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
