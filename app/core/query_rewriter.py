from app.dependencies import get_llm_client

REWRITE_PROMPT_TEMPLATE = """You are an expert search assistant.
Given the following conversation history and a new follow-up question from the user, rewrite the follow-up question into a standalone, highly-optimized search query.
The standalone query should contain all necessary context from the history so it can be used to search a vector database independently.
Do NOT answer the question. Just output the standalone query text.

Conversation History:
{history}

Follow-up Question: {question}

Standalone Query:"""

def rewrite_query(question: str, history: str) -> str:
    """
    Rewrites a follow-up question into a standalone query using the LLM
    if conversation history is present.
    """
    # If there's no history, no need to rewrite
    if not history or not history.strip():
        return question
        
    client = get_llm_client()
    prompt = REWRITE_PROMPT_TEMPLATE.format(history=history, question=question)
    
    # We use a very simple system prompt for the rewriting task
    system_prompt = "You are an expert search assistant."
    
    rewritten_query = client.generate(prompt=prompt, system=system_prompt)
    
    # Clean up output just in case the LLM wrapped it in quotes
    rewritten_query = rewritten_query.strip().strip('"').strip("'")
    
    print(f"[Query Rewriter] Original: {question}")
    print(f"[Query Rewriter] Rewritten: {rewritten_query}")
    
    return rewritten_query
      