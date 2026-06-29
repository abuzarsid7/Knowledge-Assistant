import time
import logging
from typing import Optional

try:
    from groq import Groq
except ImportError:
    Groq = None

from app.config import settings
from app.llm.base import LLMClient

logger = logging.getLogger(__name__)

class GroqClient(LLMClient):
    """
    Concrete implementation of LLMClient using the Groq API.
    Handles retries and fallback logic for timeouts or rate limits.
    """
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", max_retries: int = 3, timeout: float = 60.0):
        if Groq is None:
            raise ImportError("The 'groq' library is not installed. Please install it with 'pip install groq'.")
            
        self.client = Groq(api_key=settings.GROQ_API_KEY, timeout=timeout) if settings.GROQ_API_KEY else None
        self.model_name = model_name
        self.max_retries = max_retries
        
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generates text using the Groq API with retries for rate limits or transient errors.
        """
        if not self.client:
            raise ValueError("GROQ_API_KEY is not set in config. Cannot use Groq API.")
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
                logger.warning(f"Groq API request failed (Attempt {attempt}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    backoff_time = 2 ** attempt
                    time.sleep(backoff_time)
                    
        logger.error(f"Groq API max retries ({self.max_retries}) exceeded.")
        raise RuntimeError(f"Failed to generate response from Groq: {last_exception}") from last_exception
