import logging
from typing import Optional
from pydantic import BaseModel

from app.ingestion.ingest import run_ingestion

logger = logging.getLogger(__name__)

# --- API Schemas ---
# Defined here to maintain clean service boundaries for the FastAPI route

class IngestSummary(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    failed_documents: int
    error: Optional[str] = None

# --- Service Layer ---

def trigger_ingestion(directory: str) -> IngestSummary:
    """
    Wraps the core ingestion pipeline, providing a typed, structured summary 
    intended to be returned directly by an /ingest API endpoint.
    
    Args:
        directory: The absolute or relative path to the directory containing documents.
        
    Returns:
        IngestSummary: A Pydantic model containing the success metrics or error details.
    """
    logger.info(f"Triggering ingestion for directory: {directory}")
    
    try:
        # Call the core ingestion orchestrator
        # run_ingestion now returns a dictionary with execution metrics
        stats = run_ingestion(documents_dir=directory)
        
        return IngestSummary(
            status="success",
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            failed_documents=stats.get("failed_documents", 0)
        )
        
    except Exception as e:
        logger.exception(f"Ingestion pipeline failed catastrophically for directory: {directory}")
        return IngestSummary(
            status="error",
            total_documents=0,
            total_chunks=0,
            failed_documents=0,
            error=str(e)
        )
