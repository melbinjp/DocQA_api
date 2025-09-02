"""Minimal document loaders for the DocQA application."""
import io
from bs4 import BeautifulSoup
import PyPDF2
import docx
from .exceptions import DocumentLoaderError

def load_source(raw: bytes, ext: str) -> str:
    """
    Extracts text content from a raw byte stream based on its file extension.

    Args:
        raw: The raw bytes of the file.
        ext: The file extension (e.g., '.txt', 'pdf', '.html').

    Returns:
        The extracted text content as a string.

    Raises:
        DocumentLoaderError: If the file extension is unsupported or if there's
                             an error during parsing.
    """
    ext = ext.lower().strip('.')
    
    try:
        if ext in ['txt', 'text', 'md']:
            return raw.decode('utf-8')
        
        elif ext in ['url', 'html', 'htm']:
            html = raw.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')
            # Remove script and style tags
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            return soup.get_text(' ', strip=True)
        
        elif ext == "pdf":
            pdf_file = io.BytesIO(raw)
            reader = PyPDF2.PdfReader(pdf_file)
            texts = [page.extract_text() for page in reader.pages if page.extract_text()]
            if not texts:
                raise DocumentLoaderError("PDF text extraction resulted in no content.")
            return "\n".join(texts)
        
        elif ext == 'docx':
            doc_file = io.BytesIO(raw)
            document = docx.Document(doc_file)
            return "\n".join(p.text for p in document.paragraphs)
        
        else:
            # As a fallback, try to decode as text. If it fails, raise error.
            try:
                return raw.decode('utf-8')
            except UnicodeDecodeError:
                raise DocumentLoaderError(f"Unsupported file type: '{ext}' and could not be read as plain text.")

    except Exception as e:
        # Catch any other library-specific exceptions and wrap them.
        raise DocumentLoaderError(f"Failed to load content with extension '{ext}': {e}") from e