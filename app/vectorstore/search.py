from dataclasses import dataclass
from typing import List, Optional
from app.vectorstore.chroma_db import get_or_create_collection

# Importing from index to keep the constant central, or define here
from app.vectorstore.index import COLLECTION_NAME

@dataclass
class SearchResult:
    """
    Dataclass to standardize search results returned from the vector store.
    """
    chunk_text: str
    source: str
    page: Optional[int]
    score: float


def similarity_search(query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
    """
    Queries the Chroma vector store using a semantic embedding.
    
    Args:
        query_embedding: The embedding vector of the search query.
        top_k: The maximum number of results to return.
        
    Returns:
        A list of parsed SearchResult objects.
    """
    collection = get_or_create_collection(COLLECTION_NAME)
    
    # Chroma's query method natively supports batch queries, 
    # so we wrap our single embedding in a list.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    parsed_results = []
    
    # Safely handle empty results
    if not results or not results.get("documents") or not results["documents"][0]:
        return parsed_results
        
    # Unpack the first (and only) query's results
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    for doc, meta, distance in zip(docs, metadatas, distances):
        parsed_results.append(
            SearchResult(
                chunk_text=doc,
                source=meta.get("source", "Unknown"),
                page=meta.get("page"), # Chroma will return None if it wasn't inserted
                score=distance # Chroma's default is L2 distance (lower is closer/better)
            )
        )
        
    return parsed_results
