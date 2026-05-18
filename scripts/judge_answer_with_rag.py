# -*- coding: utf-8 -*-

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_retriever_tf import retrieve_standards
from src.judge_prompts import JUDGE_SYSTEM_PROMPT
from src.llm_client import call_dashscope
from src.json_utils import extract_json


def multiline_input(title):
    print(f"\n请输入{title}，输入完成后单独输入 END 结束：")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    print("=" * 100)
    print("基于 RAG 的 Rubric 模型回答质量评测")
    print("=" * 100)

    question = multiline_input("用户题目")
    if not question:
        print("错误: 用户题目不能为空")
        return

    answer = multiline_input("模型回答")
    if not answer:
        print("错误: 模型回答不能为空")
        return

    rubrics = multiline_input("Rubric 标准")
    if not rubrics:
        print("错误: Rubric 标准不能为空")
        return

    print("\n" + "=" * 100)
    print("正在检索相关标注标准...")
    print("=" * 100)

    retrieval_query = question + "\n" + rubrics
    retrieved_items = retrieve_standards(retrieval_query, top_k=5)

    retrieved_context = []
    for i, item in enumerate(retrieved_items, start=1):
        section = item.get("section", "")
        source_file = item.get("source_file", "")
        standard_type = item.get("standard_type", "")
        text = item.get("text", "")
        source_info = f"[来源: {source_file} | Section: {section} | Type: {standard_type}]"
        retrieved_context.append(f"{i}. {source_info}\n{text}")

    retrieved_context_str = "\n\n".join(retrieved_context)

    user_prompt = f"""【用户题目】
{question}

【模型回答】
{answer}

【Rubric 标准】
{rubrics}

【检索到的标注标准片段】
{retrieved_context_str}

请根据以上内容完成模型回答质量评测，并严格输出 JSON。"""

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    print("\n" + "=" * 100)
    print("正在调用 LLM 进行评测...")
    print("=" * 100)

    raw_output = call_dashscope(messages)

    print("\n" + "=" * 100)
    print("【模型原始输出】")
    print("=" * 100)
    print(raw_output)

    print("\n" + "=" * 100)
    print("【保存 raw_result.txt】")
    print("=" * 100)
    raw_output_path = Path(__file__).parent.parent / "raw_result.txt"
    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(raw_output)
    print(f"已保存至: {raw_output_path}")

    print("\n" + "=" * 100)
    print("【解析 JSON】")
    print("=" * 100)

    try:
        parsed_result = extract_json(raw_output)

        print("\n" + "=" * 100)
        print("【结构化评测结果】")
        print("=" * 100)
        print(json.dumps(parsed_result, ensure_ascii=False, indent=2))

        print("\n" + "=" * 100)
        print("【保存 result.json】")
        print("=" * 100)
        result_path = Path(__file__).parent.parent / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(parsed_result, f, ensure_ascii=False, indent=2)
        print(f"已保存至: {result_path}")

        print("\n" + "=" * 100)
        print("评测完成！")
        print("=" * 100)

    except Exception as e:
        print(f"JSON 解析失败，已保存 raw_result.txt")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
