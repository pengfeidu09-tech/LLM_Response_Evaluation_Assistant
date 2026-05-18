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


EXCLUDED_TYPES = ["label_taxonomy", "persona_info", "answer_evaluation", "prompt_quality"]

REQUIRED_SECTIONS = [
    "1.1 指代不清",
    "1.2 表述模糊",
    "标准过于宽泛",
    "1.3 非同对同错",
    "2.1 非原子化",
    "2.2 Rubric 之间存在冗余",
    "冗余",
    "2.3 Rubric 之间存在冲突",
    "冲突",
    "2.4 格式不规范",
    "3.1 遗漏关键约束",
    "3.2 包含无关约束",
    "4.1 Rubric 本身包含事实错误",
    "事实错误",
    "4.2 错误的推理链路"
]

EXCLUDED_SECTIONS = [
    "1.4 避免",
    "过度拟合预期解",
    "1.1.1 客观知识/计算类",
    "Rubric 的数据结构",
    "上限标准",
    "总纲：心中要有预期答案",
    "Rubric质量的区分度"
]

PREFER_KEYWORDS = [
    "原则",
    "定义",
    "判定标准",
    "核心思路",
    "必须",
    "无法",
    "不应",
    "不能"
]

AVOID_KEYWORDS = [
    "示例",
    "Case分析",
    "注意点"
]

FINAL_OUTPUT_ORDER = [
    "指代不清",
    "表述模糊",
    "非同对同错",
    "非原子化",
    "冗余",
    "冲突",
    "格式不规范",
    "遗漏关键约束",
    "包含无关约束",
    "事实错误",
    "错误的推理链路"
]


def is_valid_text(text, section):
    if not text or not text.strip():
        return False

    text_no_space = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if len(text_no_space) < 20:
        return False

    section_no_space = section.replace(" ", "").replace("\n", "").replace("\t", "") if section else ""
    if section and text_no_space == section_no_space:
        return False

    return True


def is_excluded_section(section):
    for excluded in EXCLUDED_SECTIONS:
        if excluded in section:
            return True
    return False


def get_prefer_score(text):
    score = 0
    for keyword in PREFER_KEYWORDS:
        if keyword in text:
            score += 1
    for keyword in AVOID_KEYWORDS:
        if keyword in text:
            score -= 1
    return score


def scan_metadata_for_sections(metadata):
    section_candidates = {section: None for section in REQUIRED_SECTIONS}

    for idx, meta in enumerate(metadata):
        standard_type = meta.get("standard_type", "").lower()
        if standard_type != "rubric_quality":
            continue

        section = meta.get("section", "")
        text = str(meta.get("text", "")).strip()

        if not is_valid_text(text, section):
            continue

        if is_excluded_section(section):
            continue

        for required_section in REQUIRED_SECTIONS:
            # 优先匹配 section
            section_match = required_section in section
            text_match = required_section in text

            if not section_match and not text_match:
                continue

            current = section_candidates[required_section]

            if current is None:
                section_candidates[required_section] = {
                    "idx": idx,
                    "text": text,
                    "section": section,
                    "score": 0.5,
                    "standard_type": meta.get("standard_type", ""),
                    "source_file": meta.get("source_file", ""),
                    "page_or_sheet": meta.get("page_or_sheet", ""),
                    "section_match": section_match,
                    "prefer_score": get_prefer_score(text),
                    "text_length": len(text),
                    "required_section_order": REQUIRED_SECTIONS.index(required_section)
                }
            else:
                # 优先选择 section 匹配的
                if section_match and not current.get("section_match", False):
                    section_candidates[required_section] = {
                        "idx": idx,
                        "text": text,
                        "section": section,
                        "score": 0.5,
                        "standard_type": meta.get("standard_type", ""),
                        "source_file": meta.get("source_file", ""),
                        "page_or_sheet": meta.get("page_or_sheet", ""),
                        "section_match": section_match,
                        "prefer_score": get_prefer_score(text),
                        "text_length": len(text),
                        "required_section_order": REQUIRED_SECTIONS.index(required_section)
                    }
                elif (current.get("section_match", False) == section_match):
                    # 相同匹配类型，优先选择 prefer_score 高的，其次 text 长度适中的
                    new_prefer = get_prefer_score(text)
                    old_prefer = current.get("prefer_score", 0)
                    if new_prefer > old_prefer:
                        section_candidates[required_section] = {
                            "idx": idx,
                            "text": text,
                            "section": section,
                            "score": 0.5,
                            "standard_type": meta.get("standard_type", ""),
                            "source_file": meta.get("source_file", ""),
                            "page_or_sheet": meta.get("page_or_sheet", ""),
                            "section_match": section_match,
                            "prefer_score": new_prefer,
                            "text_length": len(text),
                            "required_section_order": REQUIRED_SECTIONS.index(required_section)
                        }
                    elif new_prefer == old_prefer:
                        if len(text) > current.get("text_length", 0) and len(text) <= 1200:
                            section_candidates[required_section] = {
                                "idx": idx,
                                "text": text,
                                "section": section,
                                "score": 0.5,
                                "standard_type": meta.get("standard_type", ""),
                                "source_file": meta.get("source_file", ""),
                                "page_or_sheet": meta.get("page_or_sheet", ""),
                                "section_match": section_match,
                                "prefer_score": new_prefer,
                                "text_length": len(text),
                                "required_section_order": REQUIRED_SECTIONS.index(required_section)
                            }

    return [v for v in section_candidates.values() if v is not None]


def merge_and_deduplicate(faiss_items, rule_items):
    seen_texts = set()
    seen_sections = {}
    merged = []

    all_items = faiss_items + rule_items

    for item in all_items:
        text = item.get("text", "")
        section = item.get("section", "")

        if text in seen_texts:
            continue

        if section in seen_sections:
            existing = seen_sections[section]
            new_prefer = item.get("prefer_score", 0)
            old_prefer = existing.get("prefer_score", 0)

            if new_prefer > old_prefer:
                merged.remove(existing)
                seen_texts.discard(existing["text"])
                seen_sections[section] = item
                merged.append(item)
                seen_texts.add(text)
            elif new_prefer == old_prefer:
                if len(text) > len(existing["text"]) and len(text) <= 1200:
                    merged.remove(existing)
                    seen_texts.discard(existing["text"])
                    seen_sections[section] = item
                    merged.append(item)
                    seen_texts.add(text)
        else:
            seen_sections[section] = item
            merged.append(item)
            seen_texts.add(text)

    return merged


def sort_rubric_badcase(items):
    # 先建立 section 到 final order 的映射
    item_map = {}
    for item in items:
        section = item.get("section", "")
        text = item.get("text", "")
        content = f"{section} {text}"

        matched = False
        for idx, final_section in enumerate(FINAL_OUTPUT_ORDER):
            if final_section in content:
                if final_section not in item_map or item.get("section_match", False):
                    item_map[final_section] = {
                        **item,
                        "final_order": idx
                    }
                matched = True
                break

    # 检查缺失的 section
    for final_section in FINAL_OUTPUT_ORDER:
        if final_section not in item_map:
            print(f"[WARN] missing section: {final_section}")

    # 按 FINAL_OUTPUT_ORDER 排序
    sorted_items = []
    for final_section in FINAL_OUTPUT_ORDER:
        if final_section in item_map:
            sorted_items.append(item_map[final_section])

    return sorted_items


def get_section_display(item):
    text = item.get("text", "").strip()
    section = item.get("section", "").strip()

    if text.startswith("题目无效 | 定义"):
        return "题目无效"
    elif text.startswith("题目模糊 | 定义"):
        return "题目模糊"

    return section


def main():
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        print("Usage:")
        print('  python scripts\\rag_prompt_tf.py "题目无效和题目模糊有什么区别？"')
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

    original_query = query
    intent = None

    if "Rubric" in query and "不合格" in query:
        intent = "rubric_badcase"
        query += " 下限标准 指代不清 表述模糊 标准过于宽泛 非原子化 冗余 冲突 格式不规范 遗漏关键约束 包含无关约束 事实错误 错误推理链路"
        print(f"[INFO] 检测到 Rubric 不合格问题，intent = rubric_badcase")
        print(f"[INFO] 扩展查询: {query[:100]}...")

    top_k_raw = 60 if intent == "rubric_badcase" else 15
    print(f"[INFO] FAISS 原始召回数量: {top_k_raw}")

    query_vec = encode_query(query, tokenizer, model, device)
    scores, ids = index.search(query_vec, top_k_raw)

    faiss_items = []
    for idx, score in zip(ids[0], scores[0]):
        if int(idx) == -1:
            continue

        meta = get_meta(metadata, int(idx))
        text = str(meta.get("text", "")).strip()
        standard_type = meta.get("standard_type", "").lower()
        section = meta.get("section", "")

        if intent == "rubric_badcase":
            if standard_type in EXCLUDED_TYPES:
                continue
            if standard_type != "rubric_quality":
                continue
            if is_excluded_section(section):
                continue

        if not is_valid_text(text, section):
            continue

        faiss_items.append({
            "score": float(score),
            "text": text,
            "standard_type": meta.get("standard_type", ""),
            "source_file": meta.get("source_file", ""),
            "section": section,
            "page_or_sheet": meta.get("page_or_sheet", ""),
            "required_section_order": 999,
            "section_match": False,
            "prefer_score": get_prefer_score(text),
            "text_length": len(text)
        })

    rule_items = []
    if intent == "rubric_badcase":
        print("[INFO] 执行规则目录式补充召回...")
        rule_items = scan_metadata_for_sections(metadata)
        print(f"[INFO] 规则扫描找到 {len(rule_items)} 个候选")

    merged_items = merge_and_deduplicate(faiss_items, rule_items)

    if intent == "rubric_badcase":
        final_items = sort_rubric_badcase(merged_items)
    else:
        merged_items.sort(key=lambda x: x["score"], reverse=True)
        final_items = merged_items[:5]

    print()
    print("=" * 100)
    print("【用户问题】")
    print(original_query)
    if intent:
        print(f"【意图识别】{intent}")
    print("=" * 100)

    print()
    print("【检索到的标准片段】")
    for i, item in enumerate(final_items, start=1):
        section_display = get_section_display(item)
        print(f"\n--- 片段 {i} ---")
        print(f"Score: {round(item['score'], 4)}")
        print(f"standard_type: {item['standard_type']}")
        print(f"source_file: {item['source_file']}")
        print(f"section: {section_display}")
        print(f"page_or_sheet: {item['page_or_sheet']}")
        print("-" * 60)
        print(item["text"])

    context_parts = []
    for i, item in enumerate(final_items, start=1):
        section_display = get_section_display(item)
        source_info = f"[来源: {item['source_file']} | Section: {section_display} | Type: {item['standard_type']}]"
        context_parts.append(f"{i}. {source_info}\n{item['text']}")

    retrieved_context = "\n\n".join(context_parts)

    print()
    print("=" * 100)
    print("【给大模型的 RAG Prompt】")
    print("=" * 100)
    print()

    prompt = f"""你是大模型文本标注与 Rubric 标注标准助手。
请严格根据以下“标准片段”回答用户问题。
如果标准片段不足以支持结论，请回答“根据当前知识库无法确定”。
不要编造标准。

【用户问题】
{original_query}

【标准片段】
{retrieved_context}

请按以下结构回答：
1. 结论
2. 判断依据
3. 标注/操作建议
4. 来源依据"""

    print(prompt)


if __name__ == "__main__":
    main()
