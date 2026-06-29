import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from app.core import memory, prompts, confidence, hybrid_search, reranker, query_rewriter
from app.llm import generator
from app.services.citation_service import build_citations
from app.api.schemas import Citation

@dataclass
class RAGResponse:
    answer: str
    sources: List[Citation]
    confidence: float
    context: str = ""
    retrieval_time: float = 0.0
    generation_time: float = 0.0

def answer_question(question: str, session_id: Optional[str] = None, use_reranker: bool = True) -> RAGResponse:
    """
    The single, unified orchestrator for the RAG pipeline.
    This sequences memory retrieval, hybrid search, cross-encoder reranking,
    prompt assembly, text generation, and confidence scoring in one highly readable place.
    """
    t0 = time.time()
    
    # 1. Fetch conversation history if a session_id is provided
    history = memory.global_memory.get_history(session_id) if session_id else ""
    
    # 2. Rewrite the query based on conversation history
    search_query = query_rewriter.rewrite_query(question, history)
    
    # 3. Retrieve a broad candidate set using Hybrid Search (BM25 + Vector L2)
    candidates = hybrid_search.hybrid_retrieve(search_query, top_k=20)
    
    # 4. Rerank candidates using BM25 to get the most relevant subset
    if use_reranker:
        top_chunks = reranker.rerank(search_query, candidates, top_n=5)
    else:
        top_chunks = candidates[:5]
    
    t1 = time.time()
    
    # 4. Build the final prompt (injecting history and inline context tags)
    prompt = prompts.build_qa_prompt(question, top_chunks, history)
    
    # 5. Generate the answer using the LLM (Gemini or OpenAI)
    answer = generator.generate_answer(prompt)
    
    t2 = time.time()
    
    # 6. Compute a confidence score (penalizing for hedging language)
    retrieval_scores = [c.score for c in top_chunks]
    score = confidence.compute_confidence(retrieval_scores, answer)
    
    # 7. Extract structural citations for the final API response
    # Filter to only include chunks that are cited in the answer
    cited_chunks = []
    for i, chunk in enumerate(top_chunks, start=1):
        doc_ref = f"Document {i}"
        if doc_ref in answer or chunk.source in answer:
            cited_chunks.append(chunk)
            
    sources = build_citations(cited_chunks)
    
    # 8. Store the interaction in memory for future turns
    if session_id:
        memory.global_memory.add_turn(session_id, question, answer)
        
    # Combine chunk texts for evaluation purposes
    context_str = "\n\n".join([c.chunk_text for c in top_chunks])
        
    return RAGResponse(
        answer=answer, 
        sources=sources, 
        confidence=score, 
        context=context_str,
        retrieval_time=t1 - t0,
        generation_time=t2 - t1
    )
