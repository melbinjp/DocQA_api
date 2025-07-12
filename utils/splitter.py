"""Simple text splitter for HF Spaces"""
import re
from typing import List

def split_text(text: str, max_chars: int = 500) -> List[str]:
    """Split text into chunks"""
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return []
    
    # Split by sentences
    sentences = re.split(r'[.!?]+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks