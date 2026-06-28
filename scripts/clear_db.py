import sys
import os

# Ensure the 'app' module is discoverable from the scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    from app.vectorstore.index import clear_collection
    
    print("=== Clearing Vector Database ===")
    
    try:
        clear_collection()
        print("Successfully dropped and recreated the ChromaDB collection.")
    except Exception as e:
        print(f"Error clearing the collection: {e}")
        sys.exit(1)
