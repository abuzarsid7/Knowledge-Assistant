from app.core import prompts
from app.dependencies import get_llm_client

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
