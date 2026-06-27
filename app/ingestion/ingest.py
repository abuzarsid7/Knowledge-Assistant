import uuid
from typing import Optional

from app.utils import file_utils
from app.ingestion import loader
from app.ingestion import parser
from app.ingestion import chunker
from app.ingestion import metadata
from app.ingestion.embeddings import EmbeddingModel
from app.ingestion.chunker import Chunk

# We try to import the vectorstore instance. If it hasn't been built yet,
# we create a dummy mock so the ingestion script won't crash during development.
try:
    from app.vectorstore.index import vectorstore
except ImportError:
    class MockVectorStore:
        def add(self, chunk, embedding, meta):
            print(f"Mock add: {chunk.chunk_id}")
    vectorstore = MockVectorStore()

def run_ingestion(
    documents_dir: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50,
    embedding_model_name: str = "all-MiniLM-L6-v2"
):
    """
    Orchestrates the document ingestion pipeline.
    
    Args:
        documents_dir: The directory containing documents to ingest.
        chunk_size: Maximum token size for each chunk.
        chunk_overlap: Number of tokens to overlap between chunks.
        embedding_model_name: Name of the sentence-transformers model to use.
    """
    print(f"Starting ingestion from {documents_dir}...")
    
    # Initialize the embedding model once for the whole ingestion run
    embedding_model = EmbeddingModel(model_name=embedding_model_name)
    
    # file_utils.list_documents returns a list of Path objects
    document_paths = file_utils.list_documents(documents_dir)
    print(f"Found {len(document_paths)} documents.")
    
    for path_obj in document_paths:
        path = str(path_obj)
        print(f"Processing {path}...")
        
        # Load document (returns list of {"page_number": int|None, "raw_text": str})
        try:
            raw_pages = loader.load_document(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
            
        for page in raw_pages:
            raw_text = page.get("raw_text", "")
            if not raw_text.strip():
                continue
                
            # 1. Clean the text
            cleaned = parser.clean_text(raw_text)
            
            # 2. Extract structure (optional, if we want to use it later)
            # structure = parser.extract_structure(cleaned)
            
            # 3. Chunk the text
            # chunk_text returns a list of string chunks
            raw_chunks = chunker.chunk_text(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
            
            for index, chunk_text in enumerate(raw_chunks):
                # Construct the Chunk dataclass
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    document_name=path_obj.name,
                    page_number=page.get("page_number"),
                    chunk_index=index
                )
                
                # 4. Build metadata
                meta = metadata.build_metadata(chunk)
                
                # 5. Embed chunk
                # embed_texts expects a list and returns a list of embeddings
                embeddings_list = embedding_model.embed_texts([chunk.text])
                embedding = embeddings_list[0] if embeddings_list else []
                
                # 6. Add to vector store
                if hasattr(vectorstore, "add"):
                    vectorstore.add(chunk, embedding, meta)
                else:
                    print("vectorstore.add not implemented yet.")
                    
    print("Ingestion complete.")
