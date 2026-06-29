from typing import Optional

from app.config import settings
from app.ingestion.embeddings import EmbeddingModel
from app.llm.base import LLMClient

# --- Global Shared State ---
_embedding_model: Optional[EmbeddingModel] = None
_llm_client: Optional[LLMClient] = None

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


import logging

class FallbackLLMClient(LLMClient):
    def __init__(self, primary: LLMClient, fallback: LLMClient):
        self.primary = primary
        self.fallback = fallback
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        try:
            return self.primary.generate(prompt, system)
        except Exception as e:
            self.logger.warning(f"Primary LLM failed: {e}. Attempting fallback.")
            return self.fallback.generate(prompt, system)

def get_llm_client() -> LLMClient:
    """
    FastAPI dependency that provides the composite LLM client 
    (Groq as primary, Gemini as fallback), instantiated only once.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client
        
    from app.llm.groq_client import GroqClient
    from app.llm.gemini import GeminiClient
    
    primary = GroqClient()
    fallback = GeminiClient()
    
    _llm_client = FallbackLLMClient(primary, fallback)
    return _llm_client
