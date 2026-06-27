import copy
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np

from app.vectorstore.chroma_db import get_or_create_collection
from app.vectorstore.index import COLLECTION_NAME
from app.vectorstore.search import SearchResult
from app.core.retriever import retrieve

# Global cache for BM25 index and documents
_bm25_corpus: Optional[BM25Okapi] = None
_bm25_docs: List[Dict[str, Any]] = []

def build_bm25_index():
    """
    Fetches all chunks from Chroma and builds the BM25 index in memory.
    This can be called at startup or lazily upon the first search.
    """
    global _bm25_corpus, _bm25_docs
    
    collection = get_or_create_collection(COLLECTION_NAME)
    
    # Using collection.get() without specifying IDs returns all documents
    results = collection.get(include=["documents", "metadatas"])
    
    if not results or not results.get("documents"):
        _bm25_corpus = None
        _bm25_docs = []
        return
        
    ids = results["ids"]
    docs = results["documents"]
    metadatas = results["metadatas"]
    
    _bm25_docs = []
    tokenized_corpus = []
    
    for i in range(len(ids)):
        _bm25_docs.append({
            "chunk_id": ids[i],
            "text": docs[i],
            "meta": metadatas[i]
        })
        # Simple whitespace tokenization (can be swapped with NLTK/spaCy for advanced usage)
        tokenized_corpus.append(docs[i].lower().split())
        
    if tokenized_corpus:
        _bm25_corpus = BM25Okapi(tokenized_corpus)

def bm25_search(question: str, top_k: int = 5) -> List[SearchResult]:
    """
    Keyword search using rank_bm25 over the chunk corpus.
    """
    global _bm25_corpus, _bm25_docs
    
    if _bm25_corpus is None:
        build_bm25_index()
        
    if _bm25_corpus is None or not _bm25_docs:
        return []
        
    tokenized_query = question.lower().split()
    
    # Get scores for all documents in the corpus
    doc_scores = _bm25_corpus.get_scores(tokenized_query)
    
    # Get the indices of the top_k highest scores
    top_indices = np.argsort(doc_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        score = doc_scores[idx]
        if score <= 0:
            continue # Skip documents that don't match any keywords
            
        doc_data = _bm25_docs[idx]
        meta = doc_data["meta"]
        
        results.append(
            SearchResult(
                chunk_id=doc_data["chunk_id"],
                chunk_text=doc_data["text"],
                source=meta.get("source", "Unknown"),
                page=meta.get("page"),
                # Note: BM25 score is higher = better
                score=float(score)
            )
        )
        
    return results

def _normalize_scores(results: List[SearchResult], invert: bool = False) -> Dict[str, float]:
    """
    Normalizes scores to a 0.0 - 1.0 range based on min-max scaling.
    If invert=True, smaller original scores become closer to 1.0 (used for L2 distances).
    """
    if not results:
        return {}
        
    scores = [r.score for r in results]
    min_score, max_score = min(scores), max(scores)
    
    normalized = {}
    for r in results:
        if max_score == min_score:
            norm_score = 1.0
        else:
            norm_score = (r.score - min_score) / (max_score - min_score)
            
        if invert:
            norm_score = 1.0 - norm_score
            
        normalized[r.chunk_id] = norm_score
        
    return normalized

def hybrid_retrieve(question: str, top_k: int = 5) -> List[SearchResult]:
    """
    Merges BM25 (keyword) and Vector (semantic) results via score normalization 
    and a weighted sum, utilizing chunk_id to de-duplicate results.
    """
    # Fetch a wider net for initial search to ensure good overlap
    fetch_k = max(top_k * 2, 10)
    
    vec_results = retrieve(question, top_k=fetch_k)
    bm25_results = bm25_search(question, top_k=fetch_k)
    
    # Normalize scores (BM25: higher is better; Vector L2: lower is better)
    vec_norm = _normalize_scores(vec_results, invert=True)
    bm25_norm = _normalize_scores(bm25_results, invert=False)
    
    # Configure our hybrid weighting (can be moved to config.py)
    VEC_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    
    # Deduplicate chunks by ID
    combined_chunks: Dict[str, SearchResult] = {}
    
    for r in vec_results + bm25_results:
        if r.chunk_id not in combined_chunks:
            # Deep copy to avoid modifying the original cached SearchResult
            combined_chunks[r.chunk_id] = copy.deepcopy(r)
            
    # Compute and assign hybrid scores
    for cid, chunk in combined_chunks.items():
        v_score = vec_norm.get(cid, 0.0)
        b_score = bm25_norm.get(cid, 0.0)
        
        hybrid_score = (v_score * VEC_WEIGHT) + (b_score * BM25_WEIGHT)
        chunk.score = hybrid_score
        
    # Sort by hybrid score descending
    sorted_chunks = sorted(list(combined_chunks.values()), key=lambda x: x.score, reverse=True)
    
    return sorted_chunks[:top_k]
