"""Simple text splitter for Hugging Face Space DocQA"""
import re
from typing import List

def split_text(text: str, max_chars: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    Each chunk is up to max_chars, and overlaps the previous by `overlap` characters.
    """
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return []
    
    # Overlapping window approach
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += max_chars - overlap
    return chunks

def split_pages(pages, max_chars: int = 500, overlap: int = 100) -> List[dict]:
    """Split `[(page, text), ...]` into chunks that remember their page.

    Splitting per page rather than over one concatenated string is what makes a
    citation possible at all: once the pages are joined, no chunk can say where it
    came from. It also stops a chunk straddling a page boundary and being
    attributed to whichever page happened to come first.
    """
    out: List[dict] = []
    for page, text in pages:
        for chunk in split_text(text, max_chars=max_chars, overlap=overlap):
            out.append({"text": chunk, "page": page})
    return out
