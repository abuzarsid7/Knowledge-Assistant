from typing import Dict, List, Tuple

class ConversationMemory:
    """
    In-memory storage for conversational history.
    Stores the last N turns per session_id to provide context for follow-up questions
    (e.g., resolving pronouns like "it" to previous subjects).
    """
    
    def __init__(self, max_turns: int = 3):
        """
        Args:
            max_turns: The maximum number of conversational turns (question + answer) 
                       to remember per session. Keeping this small (e.g., 3-5) prevents 
                       blowing up the LLM context window.
        """
        self.max_turns = max_turns
        # Dictionary mapping session_id to a list of (question, answer) tuples
        self._storage: Dict[str, List[Tuple[str, str]]] = {}
        
    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """
        Records a new conversational turn for the given session.
        """
        if session_id not in self._storage:
            self._storage[session_id] = []
            
        self._storage[session_id].append((question, answer))
        
        # Enforce max turns limit by sliding the window
        if len(self._storage[session_id]) > self.max_turns:
            self._storage[session_id] = self._storage[session_id][-self.max_turns:]
            
    def get_history(self, session_id: str) -> str:
        """
        Retrieves and formats the recent conversational history for the given session.
        
        Args:
            session_id: The unique identifier for the user's session.
            
        Returns:
            A formatted string of previous questions and answers, or an empty string 
            if no history exists.
        """
        history = self._storage.get(session_id, [])
        if not history:
            return ""
            
        formatted_turns = []
        for i, (q, a) in enumerate(history, start=1):
            formatted_turns.append(f"User: {q}\nAssistant: {a}")
            
        return "Previous Conversation History:\n" + "\n---\n".join(formatted_turns) + "\n\n"
        
    def clear_session(self, session_id: str) -> None:
        """
        Clears the history for a specific session.
        """
        if session_id in self._storage:
            del self._storage[session_id]

# Export a global singleton instance for the app to share
global_memory = ConversationMemory()
