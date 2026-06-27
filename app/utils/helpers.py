import hashlib
from datetime import datetime
from typing import Optional

def generate_chunk_id(doc_name: str, page: int, index: int) -> str:
    """
    Generates a unique, reproducible ID for a document chunk.
    
    Args:
        doc_name (str): The name of the document.
        page (int): The page number the chunk belongs to.
        index (int): The sequential index of the chunk on the page or document.
        
    Returns:
        str: A unique chunk identifier (SHA-256 hash).
    """
    raw_id = f"{doc_name}_p{page}_i{index}"
    return hashlib.sha256(raw_id.encode('utf-8')).hexdigest()

def truncate_text(text: str, max_len: int) -> str:
    """
    Truncates text to a maximum length, appending '...' if truncated.
    
    Args:
        text (str): The text to truncate.
        max_len (int): The maximum allowed length.
        
    Returns:
        str: The truncated text.
    """
    if len(text) <= max_len:
        return text
    
    # Handle cases where max_len is very small
    if max_len <= 3:
        return "." * max_len
        
    return text[:max_len - 3] + "..."

def format_timestamp(ts: Optional[float] = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Formats a unix timestamp into a readable string.
    
    Args:
        ts (float, optional): The unix timestamp. Defaults to current time if None.
        fmt (str): The format string for strftime. Defaults to "YYYY-MM-DD HH:MM:SS".
        
    Returns:
        str: The formatted timestamp string.
    """
    if ts is None:
        return datetime.now().strftime(fmt)
    return datetime.fromtimestamp(ts).strftime(fmt)
