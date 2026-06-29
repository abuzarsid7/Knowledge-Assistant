from typing import List
from app.config import settings

class EmbeddingModel:
    """
    A provider-agnostic wrapper for embedding models.
    Supports both SentenceTransformers (local) and Google Gemini API (remote).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.is_gemini = model_name.startswith("models/")
        
        if self.is_gemini:
            import google.generativeai as genai
            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is not set in config.")
            genai.configure(api_key=settings.GOOGLE_API_KEY)
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Please run `pip install sentence-transformers`."
                )
            # Loads the local model into memory
            self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embed multiple texts for ingestion.
        """
        if not texts:
            return []
            
        if self.is_gemini:
            import google.generativeai as genai
            result = genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type="retrieval_document"
            )
            return result['embedding']
        else:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text for retrieval.
        """
        if self.is_gemini:
            import google.generativeai as genai
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        else:
            embedding = self.model.encode(text)
            return embedding.tolist()
