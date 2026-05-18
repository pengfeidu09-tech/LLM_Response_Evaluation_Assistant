# -*- coding: utf-8 -*-

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.rag_retriever_tf import retrieve_standards
from src.judge_prompts import JUDGE_SYSTEM_PROMPT
from src.llm_client import call_dashscope
from src.json_utils import extract_json


INPUT_FILE = "tasks.xlsx"
OUTPUT_FILE = "results_rag.xlsx"


def format_retrieved_context(items):
    context_parts = []
    for i, item in enumerate(items, start=1):
        section = item.get("section", "")
        source_file = item.get("source_file", "")
        standard_type = item.get("standard_type", "")
        text = item.get("text", "")
        source_info = f"[来源: {source_file} | Section: {section} | Type: {standard_type}]"
        context_parts.append(f"{i}. {source_info}\n{text}")
    return "\n\n".join(context_parts)


def judge_one_row(row):
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    rubrics = str(row.get("rubrics", "")).strip()

    if not question or not answer or not rubrics:
        return {
            "overall_result": "无法评测",
            "score": 0,
            "final_comment": "question / answer / rubrics 存在空值，请补充后重新评测。",
            "dimension_analysis": {},
            "rubric_check": [],
            "retrieved_standard_basis": [],
            "main_problems": [],
            "suggestions": [],
            "raw_output": "",
            "json_parse_success": False
        }

    try:
        retrieval_query = question + "\n" + rubrics
        retrieved_items = retrieve_standards(retrieval_query, top_k=5)
        retrieved_context_str = format_retrieved_context(retrieved_items)

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

        raw_output = call_dashscope(messages)

    except Exception as e:
        return {
            "overall_result": "调用失败",
            "score": 0,
            "final_comment": f"检索或 LLM 调用失败，错误信息：{e}",
            "dimension_analysis": {},
            "rubric_check": [],
            "retrieved_standard_basis": [],
            "main_problems": [],
            "suggestions": [],
            "raw_output": "",
            "json_parse_success": False
        }

    try:
        parsed = extract_json(raw_output)
        parsed["raw_output"] = raw_output
        parsed["json_parse_success"] = True
        return parsed

    except Exception as e:
        return {
            "overall_result": "JSON解析失败",
            "score": 0,
            "final_comment": f"模型输出未能成功解析为 JSON，错误信息：{e}",
            "dimension_analysis": {},
            "rubric_check": [],
            "retrieved_standard_basis": [],
            "main_problems": [],
            "suggestions": [],
            "raw_output": raw_output,
            "json_parse_success": False
        }


def main():
    input_path = Path(__file__).parent.parent / INPUT_FILE
    output_path = Path(__file__).parent.parent / OUTPUT_FILE

    if not input_path.exists():
        raise FileNotFoundError(f"没有找到 {INPUT_FILE}，请确认它位于项目根目录。")

    df = pd.read_excel(input_path, engine="openpyxl")

    required_columns = ["id", "question", "answer", "rubrics"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Excel 缺少必要列：{col}")

    results = []

    total = len(df)
    print(f"共读取到 {total} 道题，开始 RAG 批量评测...")

    for index, row in df.iterrows():
        task_id = row.get("id", index + 1)
        print(f"\n正在处理第 {index + 1}/{total} 道，ID={task_id}")

        try:
            result = judge_one_row(row)
        except Exception as e:
            print(f"处理 ID={task_id} 时出错: {e}")
            result = {
                "overall_result": "处理失败",
                "score": 0,
                "final_comment": f"处理时发生异常，错误信息：{e}",
                "dimension_analysis": {},
                "rubric_check": [],
                "retrieved_standard_basis": [],
                "main_problems": [],
                "suggestions": [],
                "raw_output": "",
                "json_parse_success": False
            }

        results.append({
            "id": task_id,
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "rubrics": row.get("rubrics", ""),

            "overall_result": result.get("overall_result", ""),
            "score": result.get("score", ""),
            "final_comment": result.get("final_comment", ""),
            "dimension_analysis": json.dumps(result.get("dimension_analysis", {}), ensure_ascii=False),
            "rubric_check": json.dumps(result.get("rubric_check", []), ensure_ascii=False),
            "retrieved_standard_basis": json.dumps(result.get("retrieved_standard_basis", []), ensure_ascii=False),
            "main_problems": json.dumps(result.get("main_problems", []), ensure_ascii=False),
            "suggestions": json.dumps(result.get("suggestions", []), ensure_ascii=False),

            "json_parse_success": result.get("json_parse_success", False),
            "raw_output": result.get("raw_output", "")
        })

        time.sleep(1)

    result_df = pd.DataFrame(results)
    result_df.to_excel(output_path, index=False)

    print(f"\n全部完成，结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
