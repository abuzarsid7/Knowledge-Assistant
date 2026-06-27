from typing import List, Dict, Any
from app.vectorstore.chroma_db import get_or_create_collection, get_client
from app.ingestion.chunker import Chunk

COLLECTION_NAME = "knowledge_base"

def add_chunks(chunks: List[Chunk], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
    """
    Batch upserts chunks, their embeddings, and metadata into the Chroma collection.
    """
    if not chunks:
        return

    collection = get_or_create_collection(COLLECTION_NAME)
    
    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

def clear_collection() -> None:
    """
    Deletes and recreates the collection to completely clear all data.
    Used by scripts/clear_db.py.
    """
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        # Ignore errors if the collection does not exist
        pass
    
    # Recreate the empty collection
    get_or_create_collection(COLLECTION_NAME)

def collection_stats() -> Dict[str, Any]:
    """
    Returns statistics about the current collection.
    Useful for a /health check endpoint.
    """
    try:
        collection = get_or_create_collection(COLLECTION_NAME)
        count = collection.count()
    except Exception:
        count = 0
        
    return {
        "collection_name": COLLECTION_NAME,
        "document_count": count
    }

class VectorStore:
    """
    A unified interface for the ingestion pipeline.
    """
    def add(self, chunk: Chunk, embedding: List[float], meta: Dict[str, Any]) -> None:
        """
        Adds a single chunk to the vector store.
        Calls the batch add_chunks function under the hood.
        """
        add_chunks(
            chunks=[chunk],
            embeddings=[embedding],
            metadatas=[meta]
        )

# Export an instance so the ingest pipeline can import it directly
vectorstore = VectorStore()
