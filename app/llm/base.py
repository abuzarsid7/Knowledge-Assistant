from abc import ABC, abstractmethod
from typing import Optional

class LLMClient(ABC):
    """
    Abstract interface for LLM providers.
    Ensures that the generator module doesn't need to know which specific
    provider (e.g., Gemini, OpenAI, Claude) is currently active.
    """
    
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generates a text response from the underlying LLM.
        
        Args:
            prompt: The main user prompt (typically includes context and question).
            system: Optional system instructions or persona for the LLM.
            
        Returns:
            The generated text string.
        """
        pass
