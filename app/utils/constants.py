# Supported file extensions for document ingestion
SUPPORTED_FILE_EXTENSIONS = [".pdf", ".docx", ".txt"]

# Prompt template identifiers
class PromptTemplates:
    QA_TEMPLATE = "qa_template"
    SUMMARIZATION_TEMPLATE = "summarization_template"
    EXTRACTION_TEMPLATE = "extraction_template"

# Confidence score bands for retrieval and reranking evaluation
class ConfidenceScoreBands:
    HIGH = 0.75
    MEDIUM = 0.50
    LOW = 0.25
