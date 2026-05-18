# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


PROJECT_ROOT = Path(__file__).parent.parent
INDEX_PATH = PROJECT_ROOT / "vector_store" / "rubric_kb.faiss"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "metadata.json"
MODEL_PATH = PROJECT_ROOT / "models" / "bge-m3"


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


class RAGRetrieverTF:
    def __init__(self):
        self.index = None
        self.metadata = None
        self.tokenizer = None
        self.model = None
        self.device = None

    def load(self):
        if self.index is None:
            self.index = faiss.read_index(str(INDEX_PATH))

        if self.metadata is None:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.metadata = data
            elif isinstance(data, dict):
                for key in ["metadata", "chunks", "items", "data"]:
                    if key in data and isinstance(data[key], list):
                        self.metadata = data[key]
                        break
                else:
                    if all(str(k).isdigit() for k in data.keys()):
                        self.metadata = [data[str(i)] for i in range(len(data)) if str(i) in data]
            if self.metadata is None:
                raise ValueError("Unsupported metadata format")

        if self.tokenizer is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
            self.model = AutoModel.from_pretrained(str(MODEL_PATH), local_files_only=True)
            self.model.to(self.device)
            self.model.eval()
            if self.device == "cuda":
                self.model.half()

    def get_meta(self, idx: int) -> Dict[str, Any]:
        if isinstance(self.metadata, list):
            return self.metadata[idx] if 0 <= idx < len(self.metadata) else {}
        if isinstance(self.metadata, dict):
            return self.metadata.get(str(idx), self.metadata.get(idx, {}))
        return {}

    def mean_pooling(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def encode_query(self, query: str) -> torch.Tensor:
        batch = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = self.model(**batch)
            emb = self.mean_pooling(outputs.last_hidden_state, batch["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu().numpy().astype("float32")

    def is_valid_text(self, text: str, section: str) -> bool:
        if not text or not text.strip():
            return False

        text_no_space = text.replace(" ", "").replace("\n", "").replace("\t", "")
        if len(text_no_space) < 20:
            return False

        section_no_space = section.replace(" ", "").replace("\n", "").replace("\t", "") if section else ""
        if section and text_no_space == section_no_space:
            return False

        return True

    def is_excluded_section(self, section: str) -> bool:
        for excluded in EXCLUDED_SECTIONS:
            if excluded in section:
                return True
        return False

    def get_prefer_score(self, text: str) -> int:
        score = 0
        for keyword in PREFER_KEYWORDS:
            if keyword in text:
                score += 1
        for keyword in AVOID_KEYWORDS:
            if keyword in text:
                score -= 1
        return score

    def scan_metadata_for_sections(self) -> List[Dict[str, Any]]:
        section_candidates = {section: None for section in REQUIRED_SECTIONS}

        for idx, meta in enumerate(self.metadata):
            standard_type = meta.get("standard_type", "").lower()
            if standard_type != "rubric_quality":
                continue

            section = meta.get("section", "")
            text = str(meta.get("text", "")).strip()

            if not self.is_valid_text(text, section):
                continue

            if self.is_excluded_section(section):
                continue

            for required_section in REQUIRED_SECTIONS:
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
                        "prefer_score": self.get_prefer_score(text),
                        "text_length": len(text),
                        "required_section_order": REQUIRED_SECTIONS.index(required_section)
                    }
                else:
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
                            "prefer_score": self.get_prefer_score(text),
                            "text_length": len(text),
                            "required_section_order": REQUIRED_SECTIONS.index(required_section)
                        }
                    elif current.get("section_match", False) == section_match:
                        new_prefer = self.get_prefer_score(text)
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

    def merge_and_deduplicate(self, faiss_items: List[Dict], rule_items: List[Dict]) -> List[Dict]:
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

    def sort_rubric_badcase(self, items: List[Dict]) -> List[Dict]:
        item_map = {}
        for item in items:
            section = item.get("section", "")
            text = item.get("text", "")
            content = f"{section} {text}"

            for idx, final_section in enumerate(FINAL_OUTPUT_ORDER):
                if final_section in content:
                    if final_section not in item_map or item.get("section_match", False):
                        item_map[final_section] = {
                            **item,
                            "final_order": idx
                        }
                    break

        for final_section in FINAL_OUTPUT_ORDER:
            if final_section not in item_map:
                print(f"[WARN] missing section: {final_section}")

        sorted_items = []
        for final_section in FINAL_OUTPUT_ORDER:
            if final_section in item_map:
                sorted_items.append(item_map[final_section])

        return sorted_items

    def retrieve_standards(self, query: str, top_k: int = 5, intent: Optional[str] = None) -> List[Dict[str, Any]]:
        self.load()

        if intent is None:
            if "Rubric" in query and "不合格" in query:
                intent = "rubric_badcase"

        original_query = query

        if intent == "rubric_badcase":
            query += " 下限标准 指代不清 表述模糊 标准过于宽泛 非原子化 冗余 冲突 格式不规范 遗漏关键约束 包含无关约束 事实错误 错误推理链路"

        top_k_raw = 60 if intent == "rubric_badcase" else 15

        query_vec = self.encode_query(query)
        scores, ids = self.index.search(query_vec, top_k_raw)

        faiss_items = []
        for idx, score in zip(ids[0], scores[0]):
            if int(idx) == -1:
                continue

            meta = self.get_meta(int(idx))
            text = str(meta.get("text", "")).strip()
            standard_type = meta.get("standard_type", "").lower()
            section = meta.get("section", "")

            if intent == "rubric_badcase":
                if standard_type in EXCLUDED_TYPES:
                    continue
                if standard_type != "rubric_quality":
                    continue
                if self.is_excluded_section(section):
                    continue

            if not self.is_valid_text(text, section):
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
                "prefer_score": self.get_prefer_score(text),
                "text_length": len(text)
            })

        rule_items = []
        if intent == "rubric_badcase":
            rule_items = self.scan_metadata_for_sections()

        merged_items = self.merge_and_deduplicate(faiss_items, rule_items)

        if intent == "rubric_badcase":
            final_items = self.sort_rubric_badcase(merged_items)
            final_top_k = top_k if top_k > 0 else 10
        else:
            merged_items.sort(key=lambda x: x["score"], reverse=True)
            final_items = merged_items
            final_top_k = top_k if top_k > 0 else 5

        result_items = final_items[:final_top_k]

        return result_items


_retriever = None


def get_retriever() -> RAGRetrieverTF:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetrieverTF()
        _retriever.load()
    return _retriever


def retrieve_standards(query: str, top_k: int = 5, intent: Optional[str] = None) -> List[Dict[str, Any]]:
    retriever = get_retriever()
    return retriever.retrieve_standards(query, top_k, intent)
