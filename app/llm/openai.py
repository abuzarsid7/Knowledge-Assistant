import time
import logging
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from app.config import settings
from app.llm.base import LLMClient

logger = logging.getLogger(__name__)

class OpenAIClient(LLMClient):
    """
    Concrete implementation of LLMClient using the OpenAI SDK.
    Serves as an optional fallback provider if Gemini is unavailable.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", max_retries: int = 3, timeout: float = 60.0):
        if OpenAI is None:
            raise ImportError("The 'openai' package is not installed. Please run `pip install openai`.")
            
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in config.")
            
        # The OpenAI SDK handles underlying connection retries natively if configured, 
        # but we set max_retries=0 internally so our outer while-loop can uniformly log 
        # identical exponential backoffs across both Gemini and OpenAI clients.
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=timeout,
            max_retries=0 
        )
        self.model_name = model_name
        self.max_retries = max_retries
        
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
            
        messages.append({"role": "user", "content": prompt})
        
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                attempt += 1
                last_exception = e
                logger.warning(f"OpenAI API request failed (Attempt {attempt}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries:
                    backoff_time = 2 ** attempt
                    time.sleep(backoff_time)
                    
        logger.error(f"OpenAI API max retries ({self.max_retries}) exceeded.")
        raise RuntimeError(f"Failed to generate response from OpenAI: {last_exception}") from last_exception
