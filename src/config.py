import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
MODELS_DIR = BASE_DIR / "models"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

BGE_M3_PATH = MODELS_DIR / "bge-m3"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "rubric_kb.faiss"
METADATA_PATH = METADATA_DIR / "metadata.json"
INDEX_INFO_PATH = VECTOR_STORE_DIR / "index_info.json"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"

DEFAULT_TOP_K = 5
