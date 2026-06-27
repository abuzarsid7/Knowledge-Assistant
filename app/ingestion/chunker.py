import re
import tiktoken
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    chunk_id: str
    text: str
    document_name: str
    page_number: int | None
    chunk_index: int

def _split_text_with_separator(text: str, separator: str, keep_separator: bool = True) -> List[str]:
    """Split text by separator and optionally keep the separator attached to the preceding text."""
    if separator == "":
        return list(text)
    
    # Use regex to split and keep separators
    # This splits into: [text1, sep, text2, sep, text3]
    splits = re.split(f"({re.escape(separator)})", text)
    
    if keep_separator:
        # Combine text with following separator
        combined = []
        for i in range(0, len(splits) - 1, 2):
            combined.append(splits[i] + splits[i+1])
        if len(splits) % 2 != 0 and splits[-1]:
            combined.append(splits[-1])
        return [c for c in combined if c]
    else:
        return [s for s in splits if s and s != separator]

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into chunks of maximum `chunk_size` tokens with `overlap` tokens.
    It attempts to split recursively on paragraph boundaries first, 
    then sentence boundaries, and finally falls back to word boundaries.
    Never splits mid-word.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    
    def get_length(s: str) -> int:
        return len(encoding.encode(s, disallowed_special=()))
        
    if not text.strip():
        return []

    # Priority of separators (from paragraph to word)
    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    
    def split_recursively(t: str, current_separators: List[str]) -> List[str]:
        """Recursively split text into atomic chunks that are smaller than chunk_size."""
        final_splits = []
        separator = current_separators[-1]
        next_separators = []
        
        for i, sep in enumerate(current_separators):
            if sep == "":
                separator = sep
                break
            if sep in t:
                separator = sep
                next_separators = current_separators[i + 1:]
                break

        splits = _split_text_with_separator(t, separator)
        
        for s in splits:
            if get_length(s) <= chunk_size:
                final_splits.append(s)
            elif next_separators:
                # If still too big, split with next granular separator
                final_splits.extend(split_recursively(s, next_separators))
            else:
                # Absolute fallback (should only hit on single words > chunk_size or if sep == "")
                final_splits.append(s)
                
        return final_splits

    # 1. Get atomic pieces (paragraphs, sentences, or words) that fit within chunk_size
    atomic_splits = split_recursively(text, separators)
    
    # 2. Pack atomic splits into overlapping chunks
    chunks = []
    current_chunk = []
    current_length = 0
    
    for split in atomic_splits:
        split_len = get_length(split)
        
        # If adding this split exceeds chunk_size (and we already have stuff in the chunk)
        if current_length + split_len > chunk_size and current_chunk:
            chunks.append("".join(current_chunk).strip())
            
            # Form overlap from the end of the current chunk
            overlap_chunk = []
            overlap_length = 0
            
            # Traverse backwards to fill overlap
            for item in reversed(current_chunk):
                item_len = get_length(item)
                if overlap_length + item_len <= overlap:
                    overlap_chunk.insert(0, item)
                    overlap_length += item_len
                else:
                    break
                    
            current_chunk = overlap_chunk
            current_length = overlap_length
            
        current_chunk.append(split)
        current_length += split_len
        
    if current_chunk:
        chunks.append("".join(current_chunk).strip())
        
    return [c for c in chunks if c]
