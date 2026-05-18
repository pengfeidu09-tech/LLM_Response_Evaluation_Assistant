# -*- coding: utf-8 -*-

print("[BOOT] search_test.py started", flush=True)

import json
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "vector_store" / "rubric_kb.faiss"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata.json"
MODEL_PATH = PROJECT_ROOT / "models" / "bge-m3"


def load_metadata(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["metadata", "chunks", "items", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

        if all(str(k).isdigit() for k in data.keys()):
            result = []
            for i in range(len(data)):
                if str(i) in data:
                    result.append(data[str(i)])
            return result

    raise ValueError("Unsupported metadata.json format.")


def get_meta(metadata, idx: int):
    if isinstance(metadata, list):
        if 0 <= idx < len(metadata):
            return metadata[idx]
        return {}

    if isinstance(metadata, dict):
        return metadata.get(str(idx), metadata.get(idx, {}))

    return {}


def print_result(rank, score, meta):
    text = str(meta.get("text", "")).replace("\n", " ").strip()
    if len(text) > 600:
        text = text[:600] + "..."

    print("\n" + "=" * 100)
    print(f"Rank: {rank}")
    print(f"Score: {score:.4f}")
    print(f"standard_type: {meta.get('standard_type', '')}")
    print(f"source_file: {meta.get('source_file', '')}")
    print(f"section: {meta.get('section', '')}")
    print(f"page_or_sheet: {meta.get('page_or_sheet', '')}")
    print("-" * 100)
    print(text)
    print("=" * 100)


def main():
    print("[CHECK] project root:", PROJECT_ROOT, flush=True)
    print("[CHECK] index path:", INDEX_PATH, flush=True)
    print("[CHECK] metadata path:", METADATA_PATH, flush=True)
    print("[CHECK] model path:", MODEL_PATH, flush=True)

    if not INDEX_PATH.exists():
        print("[ERROR] FAISS index not found.")
        return

    if not METADATA_PATH.exists():
        print("[ERROR] metadata.json not found.")
        return

    if not MODEL_PATH.exists():
        print("[ERROR] BGE-M3 model folder not found.")
        return

    print("[LOAD] loading FAISS index...", flush=True)
    index = faiss.read_index(str(INDEX_PATH))

    print("[LOAD] loading metadata...", flush=True)
    metadata = load_metadata(METADATA_PATH)

    print(f"[INFO] index.ntotal = {index.ntotal}", flush=True)
    print(f"[INFO] metadata count = {len(metadata)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LOAD] loading BGE-M3 model on {device}...", flush=True)

    model = SentenceTransformer(
        str(MODEL_PATH),
        device=device,
        trust_remote_code=True
    )

    print("\n[READY] Rubric knowledge base is ready.")
    print("[READY] Type your query after 'query>'.")
    print("[READY] Type q / quit / exit to leave.")
    print("[EXAMPLE] query> 什么是不合格 Rubric？")

    while True:
        query = input("\nquery> ").strip()

        if query.lower() in ["q", "quit", "exit"]:
            print("[EXIT] bye.")
            break

        if not query:
            continue

        try:
            top_k = 5
            query_vec = model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype("float32")

            scores, ids = index.search(query_vec, top_k)

            print(f"\n[QUERY] {query}")
            print(f"[TOP_K] {top_k}")

            for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
                if int(idx) == -1:
                    continue
                meta = get_meta(metadata, int(idx))
                print_result(rank, float(score), meta)

        except Exception as e:
            print("[ERROR] search failed:", repr(e), flush=True)


if __name__ == "__main__":
    main()
