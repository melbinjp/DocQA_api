"""Minimal document loaders for HF Spaces"""
import io
from bs4 import BeautifulSoup
import PyPDF2
import docx

def load_source(raw: bytes, ext: str) -> str:
    """Simple text extraction"""
    ext = ext.lower().strip('.')
    
    try:
        if ext in ['txt', 'text']:
            return raw.decode('utf-8', errors='ignore')
        
        elif ext in ['url', 'html']:
            html = raw.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(['script', 'style']):
                tag.decompose()
            return soup.get_text(' ', strip=True)
        
        elif ext in {".pdf", "pdf"}:
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(raw))
                texts = []
                for page in reader.pages:
                    try:
                        texts.append(page.extract_text() or "")
                    except Exception:
                        continue
                return "\n".join(texts)
            except Exception:
                return ""
        
        elif ext == 'docx':
            try:
                document = docx.Document(io.BytesIO(raw))
                return "\n".join(p.text for p in document.paragraphs)
            except Exception:
                return ""
        
        else:
            # Try as text
            return raw.decode('utf-8', errors='ignore')
            
    except Exception as e:
        return f"Error: {str(e)}"