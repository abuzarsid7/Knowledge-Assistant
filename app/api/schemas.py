from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    use_reranker: bool = True

class Citation(BaseModel):
    document: str
    page: int | None

class AskResponse(BaseModel):
    answer: str
    sources: list[Citation]
    confidence: float
    context: str = ""
    retrieval_time: float = 0.0
    

class IngestRequest(BaseModel):
    directory: str

class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    answer: str
    score: int  # 1 for upvote, 0 for downvote
