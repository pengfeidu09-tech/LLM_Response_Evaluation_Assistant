import numpy as np
from typing import List, Dict, Any
from src.vector_store import VectorStore
from src.embedder import Embedder
from src.config import DEFAULT_TOP_K


class Retriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedder = Embedder()
        self.vector_store.load()
    
    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.encode([query])
        scores, indices = self.vector_store.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.vector_store.metadata):
                result = self.vector_store.metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)
        
        return results
