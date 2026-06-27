import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.core.rag_pipeline import answer_question, RAGResponse
from app.services.citation_service import Citation

logger = logging.getLogger(__name__)

# --- API Schemas ---
# We define them here to keep the service boundaries clean, 
# or they can be moved to a dedicated schemas.py later.

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question to be answered.")
    session_id: Optional[str] = Field(None, description="Unique session ID to retrieve conversation history.")

class AskResponse(BaseModel):
    answer: str
    sources: List[Citation]
    confidence: float
    error: Optional[str] = None

# --- Service Layer ---

def handle_ask_request(request: AskRequest) -> AskResponse:
    """
    Validates the API input, delegates to the core RAG pipeline, and maps 
    the result back into the API response schema.
    
    All business logic exceptions and pipeline failures are caught here
    so the FastAPI route layer remains clean and never crashes ungracefully.
    """
    try:
        # Additional defensive validation
        question = request.question.strip()
        if not question:
            return AskResponse(
                answer="", 
                sources=[], 
                confidence=0.0, 
                error="The question cannot consist only of whitespace."
            )
            
        # Call the core RAG orchestrator
        rag_result: RAGResponse = answer_question(
            question=question, 
            session_id=request.session_id
        )
        
        # Map core result to API response
        return AskResponse(
            answer=rag_result.answer,
            sources=rag_result.sources,
            confidence=rag_result.confidence
        )
        
    except ValueError as ve:
        # Catch specific value errors (e.g. invalid configurations)
        logger.warning(f"Validation error in RAG pipeline: {ve}")
        return AskResponse(
            answer="I'm sorry, there was a configuration or validation error.", 
            sources=[], 
            confidence=0.0, 
            error=str(ve)
        )
    except Exception as e:
        # Catch all unexpected runtime exceptions (network timeouts, db failures, etc.)
        logger.exception("Unexpected error occurred while processing the ask request.")
        return AskResponse(
            answer="An internal server error occurred while processing your request.", 
            sources=[], 
            confidence=0.0, 
            error=str(e)
        )
