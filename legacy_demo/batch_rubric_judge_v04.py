#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
from pathlib import Path

import pandas as pd
import dashscope
from dotenv import load_dotenv


# ========== 1. 读取 .env 配置 ==========

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("DASHSCOPE_API_KEY")
model_name = os.getenv("MODEL_NAME", "deepseek-r1")

if not api_key:
    raise ValueError("没有读取到 DASHSCOPE_API_KEY，请检查 .env 文件。")

dashscope.api_key = api_key


# ========== 2. 文件配置 ==========

INPUT_FILE = "tasks.xlsx"
OUTPUT_FILE = "results.xlsx"


# ========== 3. System Prompt ==========

SYSTEM_PROMPT = """
你是一个严谨的大模型回答质量评测助理。

你的任务是：
根据用户题目、模型回答和 Rubric 标准，判断模型回答是否满足要求。

请你严格按照 JSON 格式输出，不要输出 Markdown，不要输出多余解释。

输出 JSON 字段如下：

{
  "overall_result": "合格 / 不合格 / 部分合格",
  "score": 0到5之间的数字,
  "dimension_analysis": {
    "instruction_following": {
      "result": "满足 / 不满足 / 部分满足",
      "reason": "理由"
    },
    "factual_correctness": {
      "result": "满足 / 不满足 / 部分满足",
      "reason": "理由"
    },
    "completeness": {
      "result": "满足 / 不满足 / 部分满足",
      "reason": "理由"
    },
    "format_quality": {
      "result": "满足 / 不满足 / 部分满足",
      "reason": "理由"
    }
  },
  "rubric_check": [
    {
      "rubric_id": 1,
      "rubric": "原始 rubric 内容",
      "result": "满足 / 不满足 / 部分满足 / 不适用",
      "reason": "判断理由"
    }
  ],
  "main_problems": [
    "主要问题1",
    "主要问题2"
  ],
  "suggestions": [
    "修改建议1",
    "修改建议2"
  ],
  "final_comment": "适合人工标注员快速参考的最终评价"
}

注意：
1. 必须逐条检查 Rubric。
2. 不要凭空补充题目没有的信息。
3. 如果模型回答没有覆盖 Rubric，要明确指出。
4. 如果某条 Rubric 不适用，要说明原因。
5. 必须输出合法 JSON。
"""


# ========== 4. JSON 提取函数 ==========

def extract_json(text):
    """
    从模型输出中提取 JSON。
    兼容 ```json ... ``` 或模型多输出了一点文字的情况。
    """
    text = str(text).strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return json.loads(text)


# ========== 5. 调用模型 ==========

def call_model(question, answer, rubrics):
    user_prompt = f"""
【用户题目】
{question}

【模型回答】
{answer}

【Rubric 标准】
{rubrics}

请根据以上内容完成模型回答质量评测。
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    response = dashscope.Generation.call(
        model=model_name,
        messages=messages,
        result_format="message"
    )

    return response.output.choices[0].message.content


# ========== 6. 处理单条数据 ==========

def judge_one_row(row):
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    rubrics = str(row.get("rubrics", "")).strip()

    if not question or not answer or not rubrics:
        return {
            "overall_result": "无法评测",
            "score": 0,
            "final_comment": "question / answer / rubrics 存在空值，请补充后重新评测。",
            "raw_output": "",
            "json_parse_success": False
        }

    raw_output = call_model(question, answer, rubrics)

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
            "raw_output": raw_output,
            "json_parse_success": False
        }


# ========== 7. 主程序：批量读取 Excel ==========

def main():
    input_path = Path(__file__).parent / INPUT_FILE
    output_path = Path(__file__).parent / OUTPUT_FILE

    if not input_path.exists():
        raise FileNotFoundError(f"没有找到 {INPUT_FILE}，请确认它和脚本在同一个文件夹。")

    df = pd.read_excel(input_path, engine="openpyxl")

    required_columns = ["id", "question", "answer", "rubrics"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Excel 缺少必要列：{col}")

    results = []

    total = len(df)
    print(f"共读取到 {total} 道题，开始批量评测...")

    for index, row in df.iterrows():
        task_id = row.get("id", index + 1)
        print(f"\n正在处理第 {index + 1}/{total} 道，ID={task_id}")

        result = judge_one_row(row)

        results.append({
            "id": task_id,
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "rubrics": row.get("rubrics", ""),

            "overall_result": result.get("overall_result", ""),
            "score": result.get("score", ""),
            "final_comment": result.get("final_comment", ""),
            "main_problems": json.dumps(result.get("main_problems", []), ensure_ascii=False),
            "suggestions": json.dumps(result.get("suggestions", []), ensure_ascii=False),
            "dimension_analysis": json.dumps(result.get("dimension_analysis", {}), ensure_ascii=False),
            "rubric_check": json.dumps(result.get("rubric_check", []), ensure_ascii=False),

            "json_parse_success": result.get("json_parse_success", False),
            "raw_output": result.get("raw_output", "")
        })

        time.sleep(1)

    result_df = pd.DataFrame(results)
    result_df.to_excel(output_path, index=False)

    print(f"\n全部完成，结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
