from typing import List
import os
import logging
from app.vectorstore.search import SearchResult

logger = logging.getLogger(__name__)

# Global cache for the CrossEncoder model
_cross_encoder_model = None

def get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        try:
            from sentence_transformers import CrossEncoder
            # We use a relatively small and fast cross-encoder by default
            _cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            logger.info("CrossEncoder model loaded successfully.")
        except ImportError:
            logger.warning("sentence-transformers is not installed. Reranking will not work.")
            return None
    return _cross_encoder_model

def rerank(question: str, candidates: List[SearchResult], top_n: int = 5) -> List[SearchResult]:
    """
    Re-scores each candidate chunk against the query using a CrossEncoder.
    Returns the top_n results sorted by CrossEncoder relevance score.
    
    Args:
        question: The user's query string.
        candidates: A larger initial set of SearchResult objects (from hybrid search).
        top_n: How many top results to return to the LLM generator.
        
    Returns:
        A list of SearchResult objects, sorted descending by relevance score.
    """
    if not candidates:
        return []
    
    model = get_cross_encoder()
    if model is None:
        # Fallback if sentence-transformers is missing
        logger.warning("Falling back to original sorting due to missing model.")
        return candidates[:top_n]
    
    # Prepare the input pairs for the cross encoder: (query, chunk_text)
    pairs = [[question, chunk.chunk_text] for chunk in candidates]
    
    try:
        # Predict scores
        scores = model.predict(pairs)
        
        # Update the score on each candidate
        for i, chunk in enumerate(candidates):
            chunk.score = float(scores[i])
            
        # Re-sort candidates descending by CrossEncoder score (higher is better)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:top_n]
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        # Fallback to initial order on failure
        return candidates[:top_n]
