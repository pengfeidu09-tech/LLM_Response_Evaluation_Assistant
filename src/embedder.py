import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from src.config import BGE_M3_PATH


class Embedder:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(BGE_M3_PATH)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)
