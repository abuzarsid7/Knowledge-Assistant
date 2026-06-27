from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from app.core import memory, prompts, confidence, hybrid_search, reranker
from app.llm import generator
from app.services import citation_service

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

def answer_question(question: str, session_id: Optional[str] = None) -> RAGResponse:
    """
    The single, unified orchestrator for the RAG pipeline.
    This sequences memory retrieval, hybrid search, cross-encoder reranking,
    prompt assembly, text generation, and confidence scoring in one highly readable place.
    """
    # 1. Fetch conversation history if a session_id is provided
    history = memory.global_memory.get_history(session_id) if session_id else ""
    
    # 2. Retrieve a broad candidate set using Hybrid Search (BM25 + Vector L2)
    candidates = hybrid_search.hybrid_retrieve(question, top_k=20)
    
    # 3. Rerank candidates using a CrossEncoder to get the most relevant subset
    top_chunks = reranker.rerank(question, candidates, top_n=5)
    
    # 4. Build the final prompt (injecting history and inline context tags)
    prompt = prompts.build_qa_prompt(question, top_chunks, history)
    
    # 5. Generate the answer using the LLM (Gemini or OpenAI)
    answer = generator.generate_answer(prompt)
    
    # 6. Compute a confidence score (penalizing for hedging language)
    retrieval_scores = [c.score for c in top_chunks]
    score = confidence.compute_confidence(retrieval_scores, answer)
    
    # 7. Extract structural citations for the final API response
    sources = citation_service.build_citations(top_chunks)
    
    # 8. Store the interaction in memory for future turns
    if session_id:
        memory.global_memory.add_turn(session_id, question, answer)
        
    return RAGResponse(answer=answer, sources=sources, confidence=score)
