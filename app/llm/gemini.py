import time
import logging
from typing import Optional
import google.generativeai as genai

from app.config import settings
from app.llm.base import LLMClient

logger = logging.getLogger(__name__)

class GeminiClient(LLMClient):
    """
    Concrete implementation of LLMClient using the Google Gemini SDK.
    Handles its own retries, backoff, and timeouts to keep the core
    generator logic clean.
    """
    
    def __init__(self, model_name: str = "gemini-1.5-pro", max_retries: int = 3, timeout: float = 60.0):
        """
        Initializes the Gemini client.
        
        Args:
            model_name: The Gemini model version to use (e.g. 'gemini-1.5-flash', 'gemini-1.5-pro')
            max_retries: Maximum number of attempts before raising an error.
            timeout: Network timeout in seconds.
        """
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in config.")
            
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generates text using the Gemini SDK with automatic retries and exponential backoff.
        """
        # We re-initialize the model object here if a system instruction is provided
        # because the system_instruction is set at the model instance level in the SDK.
        if system:
            model = genai.GenerativeModel(self.model_name, system_instruction=system)
        else:
            model = genai.GenerativeModel(self.model_name)
            
        attempt = 0
        last_exception = None
        
        while attempt < self.max_retries:
            try:
                # request_options provides a standardized way to set GRPC/REST timeouts
                response = model.generate_content(
                    prompt,
                    request_options={"timeout": self.timeout}
                )
                return response.text
            except Exception as e:
                attempt += 1
                last_exception = e
                logger.warning(f"Gemini API request failed (Attempt {attempt}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries:
                    # Exponential backoff: 2s, 4s, 8s...
                    backoff_time = 2 ** attempt
                    time.sleep(backoff_time)
                    
        logger.error(f"Gemini API max retries ({self.max_retries}) exceeded.")
        raise RuntimeError(f"Failed to generate response from Gemini: {last_exception}") from last_exception
