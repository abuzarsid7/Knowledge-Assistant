from typing import List
from app.vectorstore.search import SearchResult

SYSTEM_PROMPT = """You are an expert knowledge assistant.
You must answer the user's question using ONLY the provided context documents.
When answering, you must cite the specific sources and pages that provided the information.
If the answer is not contained within the provided context, you must explicitly state: "I don't have enough information in the provided documents."
Do NOT hallucinate, infer outside knowledge, or make up information."""

def build_qa_prompt(question: str, chunks: List[SearchResult], history: str = "") -> str:
    """
    Formats the search result chunks with source and page metadata inline,
    adds optional conversation history, and constructs the final prompt for the LLM.
    """
    context_texts = []
    for i, chunk in enumerate(chunks, start=1):
        # Format the inline metadata tags for the LLM so it can cite accurately
        page_info = f", Page: {chunk.page}" if chunk.page is not None else ""
        header = f"[Document {i} | Source: {chunk.source}{page_info}]"
        context_texts.append(f"{header}\n{chunk.chunk_text}")
        
    combined_context = "\n\n".join(context_texts)
    
    return f"""{history}Context information is provided below.
---------------------
{combined_context}
---------------------
Given the context information above and no prior knowledge, answer the following question. Make sure to cite the [Source] and [Page] for your claims.

Question: {question}
Answer:"""
