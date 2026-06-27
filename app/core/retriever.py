from typing import List

from app.config import settings
from app.ingestion.embeddings import EmbeddingModel
from app.vectorstore.search import similarity_search, SearchResult

# Singleton to avoid reloading the embedding model into memory on every query
_embedding_model = None

def get_embedding_model() -> EmbeddingModel:
    """
    Returns a singleton instance of the EmbeddingModel.
    """
    global _embedding_model
    if _embedding_model is None:
        # Depending on the config, this could be "all-MiniLM-L6-v2" or an API-based model
        _embedding_model = EmbeddingModel(model_name=settings.EMBEDDING_MODEL_NAME)
    return _embedding_model

def retrieve(question: str, top_k: int = 5) -> List[SearchResult]:
    """
    Plain semantic retriever.
    Embeds the user query and calls the vector store's similarity search.
    
    Note: Hybrid search or reranking pipelines should wrap around this function, 
    rather than replacing it.
    
    Args:
        question: The user's query text.
        top_k: The number of semantic results to retrieve.
        
    Returns:
        A list of parsed SearchResult objects containing text, source, and score.
    """
    # 1. Embed the query
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(question)
    
    # 2. Retrieve nearest neighbors from the vector store
    results = similarity_search(query_embedding=query_embedding, top_k=top_k)
    
    return results
