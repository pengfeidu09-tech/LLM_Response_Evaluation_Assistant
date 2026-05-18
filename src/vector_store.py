import faiss
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from src.config import FAISS_INDEX_PATH, METADATA_PATH, INDEX_INFO_PATH


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []
        self.index_info = {}
    
    def build_index(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]], model_path: str):
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = metadata_list
        self.index_info = {
            "model_path": model_path,
            "embedding_dim": dim,
            "total_chunks": len(metadata_list),
            "created_at": datetime.now().isoformat(),
            "index_type": "IndexFlatIP"
        }
    
    def save(self):
        FAISS_INDEX_PATH.parent.mkdir(exist_ok=True, parents=True)
        METADATA_PATH.parent.mkdir(exist_ok=True, parents=True)
        INDEX_INFO_PATH.parent.mkdir(exist_ok=True, parents=True)
        
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        with open(INDEX_INFO_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.index_info, f, ensure_ascii=False, indent=2)
    
    def load(self):
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        with open(INDEX_INFO_PATH, 'r', encoding='utf-8') as f:
            self.index_info = json.load(f)
