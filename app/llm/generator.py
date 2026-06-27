from typing import List

from app.config import settings
from app.llm.base import LLMClient
from app.llm.gemini import GeminiClient
from app.llm.openai import OpenAIClient
from app.vectorstore.search import SearchResult
from app.core import prompts

# Store a singleton instance so we don't re-initialize clients repeatedly
_client_instance = None

def get_llm_client() -> LLMClient:
    """
    Factory function that reads the configuration and returns the 
    appropriate initialized LLM client (Gemini or OpenAI).
    """
    global _client_instance
    if _client_instance is not None:
        return _client_instance
        
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "gemini":
        _client_instance = GeminiClient()
    elif provider == "openai":
        _client_instance = OpenAIClient()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Must be 'gemini' or 'openai'.")
        
    return _client_instance

def generate_answer(question: str, context_chunks: List[SearchResult]) -> str:
    """
    Builds the final prompt using the retrieved context chunks and the user's question,
    and calls the configured LLM client to generate the answer.
    
    Args:
        question: The user's query.
        context_chunks: The list of parsed SearchResult objects from the vector store.
        
    Returns:
        The generated text string from the LLM.
    """
    client = get_llm_client()
    
    # 1. Format the context chunks into a single readable block
    context_texts = []
    for chunk in context_chunks:
        # Add source and page info directly to the chunk string so the LLM can see it and cite it
        source_header = f"--- Source: {chunk.source}"
        if chunk.page is not None:
            source_header += f", Page: {chunk.page}"
        source_header += " ---"
        
        context_texts.append(f"{source_header}\n{chunk.chunk_text}")
        
    combined_context = "\n\n".join(context_texts)
    
    # 2. Build the final prompt string using our templates
    prompt = prompts.build_rag_prompt(question, combined_context)
    
    # 3. Call the LLM client
    answer = client.generate(
        prompt=prompt,
        system=prompts.SYSTEM_PROMPT
    )
    
    return answer
