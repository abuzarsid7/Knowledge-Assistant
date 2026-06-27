import uuid
from typing import Optional, Dict, Any

from app.utils import file_utils
from app.ingestion import loader
from app.ingestion import parser
from app.ingestion import chunker
from app.ingestion import metadata
from app.ingestion.embeddings import EmbeddingModel
from app.ingestion.chunker import Chunk

# Try to import the actual vectorstore. Fallback to mock for testing/initial dev.
try:
    from app.vectorstore.index import add_chunks
except ImportError:
    def add_chunks(chunks, embeddings, metadatas):
        print(f"Mock add_chunks: {len(chunks)} chunks")

def run_ingestion(
    documents_dir: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Dict[str, Any]:
    """
    Orchestrates the document ingestion pipeline.
    
    Args:
        documents_dir: The directory containing documents to ingest.
        chunk_size: Maximum token size for each chunk.
        chunk_overlap: Number of tokens to overlap between chunks.
        embedding_model_name: Name of the sentence-transformers model to use.
        
    Returns:
        Dict containing ingestion statistics (documents processed, chunks created, failures).
    """
    print(f"Starting ingestion from {documents_dir}...")
    
    embedding_model = EmbeddingModel(model_name=embedding_model_name)
    document_paths = file_utils.list_documents(documents_dir)
    
    stats = {
        "total_documents": len(document_paths),
        "total_chunks": 0,
        "failed_documents": 0
    }
    
    for path_obj in document_paths:
        path = str(path_obj)
        print(f"Processing {path}...")
        
        try:
            raw_pages = loader.load_document(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            stats["failed_documents"] += 1
            continue
            
        chunks_to_add = []
        embeddings_to_add = []
        metadatas_to_add = []
            
        for page in raw_pages:
            raw_text = page.get("raw_text", "")
            if not raw_text.strip():
                continue
                
            # Clean text
            cleaned = parser.clean_text(raw_text)
            
            # Chunk the text
            raw_chunks = chunker.chunk_text(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
            
            for index, chunk_text in enumerate(raw_chunks):
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_text,
                    document_name=path_obj.name,
                    page_number=page.get("page_number"),
                    chunk_index=index
                )
                
                meta = metadata.build_metadata(chunk)
                
                # We can batch embed later, but for simplicity embed individually here
                embeddings_list = embedding_model.embed_texts([chunk.text])
                embedding = embeddings_list[0] if embeddings_list else []
                
                chunks_to_add.append(chunk)
                embeddings_to_add.append(embedding)
                metadatas_to_add.append(meta)
                
                stats["total_chunks"] += 1
                
        # Batch write to vector store
        if chunks_to_add:
            add_chunks(chunks_to_add, embeddings_to_add, metadatas_to_add)
            
    print("Ingestion complete.")
    return stats
