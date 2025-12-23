"""
settings.py

Central configuration for paths and chunking parameters used by
the vector pipeline (ingestion + retrieval).
"""
from pathlib import Path

# Directory: /CHATBOT/multi_turn_pipeline
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT: Path = _THIS_DIR.parent

# Where to persist the Chroma DB.
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"

# Optional: zipped Chroma DB archive (for distribution via Git).
CHROMA_ARCHIVE: Path = PROJECT_ROOT / "chroma_db.zip"

# Path to chat history SQLite database
CHAT_HISTORY_DB_PATH: Path = PROJECT_ROOT / "chat_history.db"

# Embedding model (bi-encoder)
EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"

# Cross-encoder reranker model
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

# LLM model for RAG
CLOUD_LLM_MODEL_NAME: str = "google/gemma-3-27b-it:free",
#CLOUD_LLM_MODEL_NAME: str = "google/gemma-3-12b-it:free",

# Path to OpenRouter API key
OPENROUTER_API_KEY_PATH: str = PROJECT_ROOT / ".env"