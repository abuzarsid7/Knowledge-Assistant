from typing import Dict, Any
from app.ingestion.chunker import Chunk

def build_metadata(chunk: Chunk, **kwargs) -> Dict[str, Any]:
    """
    Converts a Chunk into the metadata dictionary ChromaDB expects.
    
    Args:
        chunk: The Chunk dataclass instance.
        **kwargs: Additional document-level tags to add (e.g., category="HR").
                  Values must be str, int, float, or bool.
                  
    Returns:
        A dictionary containing the metadata.
    """
    metadata: Dict[str, Any] = {
        "source": chunk.document_name,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index
    }
    
    # ChromaDB metadata values cannot be None, so only include page if it exists
    if chunk.page_number is not None:
        metadata["page"] = chunk.page_number
        
    # Apply any additional document-level tags passed in
    # (e.g. if we later want to add document categories)
    for key, value in kwargs.items():
        if value is not None:
            metadata[key] = value
            
    return metadata
