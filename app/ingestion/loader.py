import os
import pypdf
import docx

def load_pdf(path: str) -> list[dict]:
    """
    Reads a PDF file into memory.
    Returns [{"page_number": int, "raw_text": str}] per page.
    """
    pages = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            # extract_text can return None or empty string, we should handle it
            if text is not None:
                pages.append({
                    "page_number": i + 1,
                    "raw_text": text
                })
    return pages

def load_docx(path: str) -> list[dict]:
    """
    Reads a DOCX file into memory.
    Since docx has no native "pages," treat the whole doc as page_number = None.
    """
    doc = docx.Document(path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    
    text = "\n".join(full_text)
    
    return [{
        "page_number": None,
        "raw_text": text
    }]

def load_txt(path: str) -> list[dict]:
    """
    Reads a TXT file into memory.
    Treats the whole document as a single page.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        
    return [{
        "page_number": None,
        "raw_text": text
    }]

def load_document(path: str) -> list[dict]:
    """
    Dispatcher that picks the right loader by extension.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext == ".txt":
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
