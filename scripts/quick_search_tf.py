# -*- coding: utf-8 -*-

import sys
import json
from pathlib import Path

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "vector_store" / "rubric_kb.faiss"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata.json"
MODEL_PATH = PROJECT_ROOT / "models" / "bge-m3"


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["metadata", "chunks", "items", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]

        if all(str(k).isdigit() for k in data.keys()):
            return [data[str(i)] for i in range(len(data)) if str(i) in data]

    raise ValueError("Unsupported metadata format")


def get_meta(metadata, idx):
    if isinstance(metadata, list):
        return metadata[idx] if 0 <= idx < len(metadata) else {}

    if isinstance(metadata, dict):
        return metadata.get(str(idx), metadata.get(idx, {}))

    return {}


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_query(query, tokenizer, model, device):
    batch = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)
        emb = mean_pooling(outputs.last_hidden_state, batch["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)

    return emb.cpu().numpy().astype("float32")


def main():
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        print("Usage:")
        print('  python scripts\\quick_search_tf.py "什么是不合格 Rubric？"')
        return

    print("[LOAD] FAISS index:", INDEX_PATH, flush=True)
    index = faiss.read_index(str(INDEX_PATH))

    print("[LOAD] metadata:", METADATA_PATH, flush=True)
    metadata = load_metadata(METADATA_PATH)

    print("[INFO] index.ntotal =", index.ntotal, flush=True)
    print("[INFO] metadata count =", len(metadata), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[LOAD] device =", device, flush=True)

    print("[LOAD] tokenizer/model:", MODEL_PATH, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model = AutoModel.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model.to(device)
    model.eval()

    if device == "cuda":
        model.half()

    query_vec = encode_query(query, tokenizer, model, device)
    scores, ids = index.search(query_vec, 5)

    print()
    print("[QUERY]", query)
    print("[RESULTS]")

    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if int(idx) == -1:
            continue

        meta = get_meta(metadata, int(idx))
        text = str(meta.get("text", "")).replace("\n", " ").strip()

        if len(text) > 600:
            text = text[:600] + "..."

        print()
        print("=" * 100)
        print("Rank:", rank)
        print("Score:", round(float(score), 4))
        print("standard_type:", meta.get("standard_type", ""))
        print("source_file:", meta.get("source_file", ""))
        print("section:", meta.get("section", ""))
        print("page_or_sheet:", meta.get("page_or_sheet", ""))
        print("-" * 100)
        print(text)
        print("=" * 100)


if __name__ == "__main__":
    main()
