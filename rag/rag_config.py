from typing import Dict, Any

try:
    from config import LLM_MODEL
except ImportError:
    LLM_MODEL = "gpt-4o"  # Default fallback

# RAG system configuration
RAG_CONFIG: Dict[str, Any] = {
    # Document processing
    "chunk_size": 1000,
    "chunk_overlap": 200,

    # Retrieval
    "top_k": 5,

    # Models
    "embedding_model_name": "openai",
    "llm_provider": "openai",
    "llm_model_name": LLM_MODEL,

    # Database
    "db_name": "supabase",

    # Processing
    "batch_size": 100,

    # Table names
    "documents_table": "documents",
}

# System prompts
SYSTEM_PROMPTS = {
    "summarization": """You are a summarization engine for a question-answering system focused on reducing 
car accidents. Your task is to read through a provided text (extracted from PDF files) and generate a concise summary 
of the specified text, focusing only on relevant information for understanding accident trends. 
Don't add any details that are not specified directly in the text itself.
"""
}