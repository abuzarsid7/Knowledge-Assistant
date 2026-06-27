from typing import List

from app.config import settings
from app.llm.base import LLMClient
from app.llm.gemini import GeminiClient
from app.llm.openai import OpenAIClient
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

def generate_answer(prompt: str) -> str:
    """
    Calls the configured LLM client to generate the answer based on a pre-built prompt.
    """
    client = get_llm_client()
    
    answer = client.generate(
        prompt=prompt,
        system=prompts.SYSTEM_PROMPT
    )
    
    return answer
