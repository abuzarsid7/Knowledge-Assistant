import chromadb
from chromadb.config import Settings
from app.config import settings

_client_instance = None

def get_client() -> chromadb.ClientAPI:
    """
    Returns a singleton instance of the Chroma PersistentClient.
    This points to the persistence directory specified in the config.
    Using a persistent client ensures that embeddings are saved to disk.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
    return _client_instance


def get_or_create_collection(name: str) -> chromadb.Collection:
    """
    Retrieves a ChromaDB collection by name, or creates it if it doesn't exist.
    """
    client = get_client()
    # If using your own embedding function from app/ingestion/embeddings.py
    # you can pass embedding_function=None to chroma since you pass the raw embeddings manually
    return client.get_or_create_collection(name=name)
