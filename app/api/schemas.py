from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None

class Citation(BaseModel):
    document: str
    page: int | None

class AskResponse(BaseModel):
    answer: str
    sources: list[Citation]
    confidence: float

class IngestRequest(BaseModel):
    directory: str
