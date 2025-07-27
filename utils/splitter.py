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