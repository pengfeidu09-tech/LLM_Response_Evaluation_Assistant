#!/usr/bin/env python
# coding: utf-8

import os
import json
from pathlib import Path

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


# ========== 2. 工具函数：支持多行输入 ==========

def multiline_input(title):
    """
    支持多行输入。
    输入 END 后结束。
    """
    print(f"\n请输入{title}，输入完成后单独输入 END 结束：")

    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    return "\n".join(lines).strip()


# ========== 3. 工具函数：从模型输出中提取 JSON ==========

def extract_json(text):
    """
    尝试从模型输出中提取 JSON。
    防止模型输出 ```json ... ``` 这种格式。
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    elif text.startswith("```"):
        text = text.replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return json.loads(text)


# ========== 4. 调用模型 ==========

def get_response(messages):
    response = dashscope.Generation.call(
        model=model_name,
        messages=messages,
        result_format="message"
    )
    return response


# ========== 5. 核心函数：评测模型回答 ==========

def judge_answer(user_question, model_answer, rubrics):
    system_prompt = """
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

    user_prompt = f"""
【用户题目】
{user_question}

【模型回答】
{model_answer}

【Rubric 标准】
{rubrics}

请根据以上内容完成模型回答质量评测。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = get_response(messages)
    raw_output = response.output.choices[0].message.content

    return raw_output


# ========== 6. 主程序 ==========

if __name__ == "__main__":
    print("=== 基于 Rubric 的模型回答质量评测助手 v0.3 ===")

    user_question = multiline_input("用户题目")
    model_answer = multiline_input("模型回答")
    rubrics = multiline_input("Rubric 标准")

    print("\n正在调用模型评测，请稍等...\n")

    raw_result = judge_answer(user_question, model_answer, rubrics)

    print("=== 模型原始输出 ===")
    print(raw_result)

    print("\n=== 尝试解析 JSON ===")

    try:
        parsed_result = extract_json(raw_result)

        print(json.dumps(parsed_result, ensure_ascii=False, indent=2))

        output_path = Path(__file__).parent / "result.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed_result, f, ensure_ascii=False, indent=2)

        print(f"\n评测结果已保存到：{output_path}")

    except Exception as e:
        print("JSON 解析失败。模型可能没有严格输出 JSON。")
        print("错误信息：", e)

        raw_output_path = Path(__file__).parent / "raw_result.txt"

        with open(raw_output_path, "w", encoding="utf-8") as f:
            f.write(raw_result)

        print(f"原始结果已保存到：{raw_output_path}")
