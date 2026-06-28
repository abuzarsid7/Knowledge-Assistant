from typing import Optional

from app.config import settings
from app.ingestion.embeddings import EmbeddingModel
from app.llm.base import LLMClient

# --- Global Shared State ---
_embedding_model: Optional[EmbeddingModel] = None
_llm_client: Optional[LLMClient] = None
_reranker_model = None

# --- FastAPI Dependency Providers ---

def get_embedding_model() -> EmbeddingModel:
    """
    FastAPI dependency that lazy-loads and returns a shared EmbeddingModel instance.
    Prevents reloading heavy transformer weights into memory on every request.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(model_name=settings.EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_reranker_model():
    """
    FastAPI dependency that lazy-loads and returns a shared CrossEncoder instance.
    Prevents reloading the reranker weights on every request.
    """
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def get_llm_client() -> LLMClient:
    """
    FastAPI dependency that provides the correct LLM client (Gemini/OpenAI) 
    based on configuration, instantiated only once per app lifecycle.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client
        
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "gemini":
        from app.llm.gemini import GeminiClient
        _llm_client = GeminiClient()
    elif provider == "openai":
        from app.llm.openai import OpenAIClient
        _llm_client = OpenAIClient()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Must be 'gemini' or 'openai'.")
        
    return _llm_client
