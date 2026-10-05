from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Chroma database
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

# Collection name
COLLECTION_NAME = "university_knowledge"

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval settings
TOP_K = 3

# Chunking settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120