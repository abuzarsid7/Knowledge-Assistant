import sys
import os

# Ensure the 'app' module is discoverable from the scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    from app.vectorstore.index import clear_collection
    from app.ingestion.ingest import run_ingestion
    
    print("=== Starting Full Index Rebuild ===")
    
    print("\n1. Clearing existing ChromaDB collection...")
    try:
        clear_collection()
        print("Collection cleared successfully.")
    except Exception as e:
        print(f"Error clearing collection: {e}")
        sys.exit(1)
        
    print("\n2. Re-running ingestion pipeline...")
    try:
        stats = run_ingestion("documents/")
        print("\n=== Rebuild Complete ===")
        print(f"Total Documents Processed: {stats.get('total_documents', 0)}")
        print(f"Total Chunks Generated: {stats.get('total_chunks', 0)}")
        print(f"Failed Documents: {stats.get('failed_documents', 0)}")
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        sys.exit(1)
