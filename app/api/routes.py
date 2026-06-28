from fastapi import APIRouter

from app.api.schemas import AskRequest, AskResponse, IngestRequest
from app.services.ask_service import handle_ask_request
from app.services.ingest_service import trigger_ingestion, IngestSummary

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    """
    Answers a question based on the ingested document knowledge base.
    Delegates all logic to ask_service.py to keep the route thin.
    """
    return handle_ask_request(request)

@router.post("/ingest", response_model=IngestSummary)
def ingest_endpoint(request: IngestRequest):
    """
    Triggers the document ingestion pipeline.
    Delegates all logic to ingest_service.py to keep the route thin.
    """
    return trigger_ingestion(request.directory)
