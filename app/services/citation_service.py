from typing import List
from app.api.schemas import Citation
from app.vectorstore.search import SearchResult

def build_citations(chunks: List[SearchResult]) -> List[Citation]:
    """
    Extracts citation metadata from the final chunks for the API response.
    Deduplicates multiple chunks originating from the same document/page 
    into a single citation entry.
    """
    citations = []
    seen = set()
    
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    document=chunk.source,
                    page=chunk.page
                )
            )
            
    return citations
