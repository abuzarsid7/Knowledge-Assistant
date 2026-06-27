import hashlib
from pathlib import Path

# Common document extensions to look for
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md", ".docx", ".csv"}

def list_documents(directory: str) -> list[Path]:
    """
    Scans the given directory for supported document extensions.
    
    Args:
        directory (str): The directory path to scan.
        
    Returns:
        list[Path]: A list of Path objects for all matching documents.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    
    documents = []
    # Using rglob to search recursively
    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(file_path)
            
    return documents

def get_file_hash(path: Path) -> str:
    """
    Calculates the SHA-256 hash of a file.
    Useful for detecting whether a document changed since last ingestion.
    
    Args:
        path (Path): Path to the file.
        
    Returns:
        str: The hexadecimal SHA-256 hash of the file.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def ensure_dir(path: Path) -> None:
    """
    Ensures that a directory exists, creating it and its parents if necessary.
    
    Args:
        path (Path): The directory path to ensure.
    """
    path.mkdir(parents=True, exist_ok=True)
