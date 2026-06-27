from typing import List

class EmbeddingModel:
    """
    A provider-agnostic wrapper for embedding models.
    This encapsulates the embedding logic so that if you ever swap to OpenAI 
    or another provider, you only need to change this file.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Loads the model once during initialization.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please run `pip install sentence-transformers`."
            )
            
        self.model_name = model_name
        # Loads the model into memory
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embed multiple texts for ingestion.
        
        Args:
            texts: A list of strings to embed.
            
        Returns:
            A list of embedding vectors (list of lists of floats).
        """
        if not texts:
            return []
            
        # sentence-transformers returns a numpy array, we convert to a standard
        # Python list to keep the interface provider-agnostic and JSON serializable.
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text for retrieval.
        Using a separate method makes the call site intent clear.
        
        Args:
            text: A single string to embed.
            
        Returns:
            A single embedding vector (list of floats).
        """
        embedding = self.model.encode(text)
        return embedding.tolist()
