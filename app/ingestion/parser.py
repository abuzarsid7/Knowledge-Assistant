import re
from typing import Dict

def clean_text(text: str) -> str:
    """
    Clean the extracted text by removing excess whitespace, 
    fixing broken hyphenation, and removing common headers/footers.
    """
    if not text:
        return ""
        
    # Fix broken hyphenation (e.g., "word-\nword" -> "wordword")
    text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text)
    
    # Remove common page numbers / footers like "Page 1", "12 / 14", "- 3 -" on their own lines
    text = re.sub(r'(?im)^(?:\s*(?:page|p\.)\s*\d+(?:\s*(?:of|/)\s*\d+)?\s*)$', '', text)
    text = re.sub(r'(?im)^(?:\s*-\s*\d+\s*-\s*)$', '', text)
    
    # Strip excess whitespace
    # Replace 3 or more newlines with exactly 2 newlines (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace 2 or more spaces or tabs with a single space
    text = re.sub(r'[ \t]{2,}', ' ', text)
    
    # Trim leading and trailing whitespace
    return text.strip()


def extract_structure(text: str) -> Dict[str, str]:
    """
    Extract structure from text, such as detecting section headings.
    Returns a dictionary mapping section headings to their content.
    """
    if not text:
        return {}
        
    structure: Dict[str, str] = {}
    
    lines = text.split('\n')
    current_heading = "Root"
    current_content = []
    
    # Simple regex to match potential headings:
    # 1. Markdown style: "# Heading"
    # 2. Numbered style: "1. Introduction" or "1.1. Background"
    # 3. All caps style: "INTRODUCTION"
    heading_pattern = re.compile(
        r'^(#{1,6}\s+.+|'             # Markdown
        r'\d+(?:\.\d+)*\.\s+[A-Z].+|' # Numbered (e.g., "1. Intro", "1.1. Background")
        r'[A-Z][A-Z\s\d]+:?)$'        # All caps (can optionally end with a colon)
    )
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_content:
                current_content.append(line)
            continue
            
        # Check if line looks like a heading (heuristic: < 100 chars and matches pattern)
        if len(stripped) < 100 and heading_pattern.match(stripped):
            # Save the previous section
            if current_heading not in structure:
                structure[current_heading] = '\n'.join(current_content).strip()
            else:
                # In case of duplicate headings, append
                structure[current_heading] += '\n' + '\n'.join(current_content).strip()
                
            current_heading = stripped
            current_content = []
        else:
            current_content.append(line)
            
    # Save the last section
    if current_heading not in structure:
        structure[current_heading] = '\n'.join(current_content).strip()
    else:
        structure[current_heading] += '\n' + '\n'.join(current_content).strip()
        
    # Clean up empty sections if needed, though sometimes 'Root' might just be empty
    return structure
