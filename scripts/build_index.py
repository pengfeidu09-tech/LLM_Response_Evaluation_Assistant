import sys
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CHUNKS_PATH, BGE_M3_PATH
from src.embedder import Embedder
from src.vector_store import VectorStore


def main():
    if not CHUNKS_PATH.exists():
        print(f"错误: 未找到 {CHUNKS_PATH}，请先运行 extract_documents.py")
        return
    
    print("加载 chunks...")
    chunks = []
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    
    if not chunks:
        print("错误: 没有有效的 chunks")
        return
    
    texts = [c["text"] for c in chunks]
    
    print(f"加载 BGE-M3 模型 (path: {BGE_M3_PATH})...")
    embedder = Embedder(str(BGE_M3_PATH))
    
    print("生成 embeddings...")
    embeddings = embedder.encode(texts)
    
    print("构建 FAISS 索引...")
    vector_store = VectorStore()
    vector_store.build_index(embeddings, chunks, str(BGE_M3_PATH))
    
    print("保存索引和元数据...")
    vector_store.save()
    
    print(f"\n索引构建完成！")
    print(f"Embedding 维度: {embeddings.shape[1]}")
    print(f"Chunk 数量: {len(chunks)}")
    print(f"索引保存路径: {vector_store.index_info['created_at']}")


if __name__ == "__main__":
    main()
