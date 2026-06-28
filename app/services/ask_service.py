import logging
from fastapi import HTTPException

from app.core.rag_pipeline import answer_question, RAGResponse
from app.api.schemas import AskRequest, AskResponse

logger = logging.getLogger(__name__)

def handle_ask_request(request: AskRequest) -> AskResponse:
    """
    Validates the API input, delegates to the core RAG pipeline, and maps 
    the result back into the API response schema.
    
    All business logic exceptions and pipeline failures are caught here and
    translated to FastAPI HTTPExceptions, ensuring the route layer remains clean.
    """
    try:
        # Additional defensive validation
        question = request.question.strip()
        if not question:
            raise ValueError("The question cannot consist only of whitespace.")
            
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
        # Catch specific value errors (e.g. invalid configurations or empty inputs)
        logger.warning(f"Validation error in RAG pipeline: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Catch all unexpected runtime exceptions (network timeouts, db failures, etc.)
        logger.exception("Unexpected error occurred while processing the ask request.")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing your request.")
