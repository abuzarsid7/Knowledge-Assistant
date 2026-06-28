from typing import List
from app.vectorstore.search import SearchResult
from app.dependencies import get_reranker_model

def rerank(question: str, candidates: List[SearchResult], top_n: int = 5) -> List[SearchResult]:
    """
    Scores each (question, chunk) pair using a CrossEncoder.
    Returns the top_n re-sorted candidates based on semantic relevance.
    
    Args:
        question: The user's query string.
        candidates: A larger initial set of SearchResult objects (e.g., from hybrid_retrieve).
        top_n: How many top results to return to the LLM generator.
        
    Returns:
        A list of SearchResult objects, sorted descending by relevance score.
    """
    if not candidates:
        return []
        
    model = get_reranker_model()
    
    # The CrossEncoder expects a list of sentence pairs: [[query, doc1], [query, doc2], ...]
    pairs = [[question, chunk.chunk_text] for chunk in candidates]
    
    # Compute the relevance scores (returns a numpy array of floats)
    scores = model.predict(pairs)
    
    # Update the score property on the candidate chunks
    for i, chunk in enumerate(candidates):
        chunk.score = float(scores[i])
        
    # Re-sort candidates descending by their new CrossEncoder score
    candidates.sort(key=lambda x: x.score, reverse=True)
    
    return candidates[:top_n]
