from fastapi import APIRouter
from app.vectorstore.index import collection_stats

router = APIRouter()

@router.get("/health")
def health_check():
    """
    Returns the basic health status of the application along with the 
    underlying vector store collection stats. This is useful for quickly 
    verifying if the database has been populated or if it is empty.
    """
    stats = collection_stats()
    
    return {
        "status": "healthy",
        "vectorstore": stats
    }
