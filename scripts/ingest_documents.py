import sys
import os

# Ensure the 'app' module is discoverable from the scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    from app.ingestion.ingest import run_ingestion
    
    print("=== Starting CLI Ingestion ===")
    
    # Trigger the ingestion process for the default 'documents/' directory
    stats = run_ingestion("documents/")
    
    print("\n=== Ingestion Summary ===")
    print(f"Total Documents Processed: {stats.get('total_documents', 0)}")
    print(f"Total Chunks Generated: {stats.get('total_chunks', 0)}")
    print(f"Failed Documents: {stats.get('failed_documents', 0)}")
