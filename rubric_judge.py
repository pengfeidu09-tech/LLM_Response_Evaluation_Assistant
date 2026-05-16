#!/usr/bin/env python
# coding: utf-8

import os
import dashscope
from dotenv import load_dotenv
from pathlib import Path


# 读取当前目录下的 .env 文件
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


# 从 .env 或系统环境变量中读取 API Key 和模型名
api_key = os.getenv("DASHSCOPE_API_KEY")
model_name = os.getenv("MODEL_NAME", "deepseek-r1")


if not api_key:
    raise ValueError("没有读取到 DASHSCOPE_API_KEY，请检查 .env 文件是否正确配置。")


dashscope.api_key = api_key


def get_response(messages):
    """
    调用大模型接口，返回模型回答
    """
    response = dashscope.Generation.call(
        model=model_name,
        messages=messages,
        result_format="message"
    )
    return response


def judge_answer(user_question, model_answer):
    """
    Rubric 评测助手：
    输入用户问题和模型回答，让大模型判断回答质量。
    """

    system_prompt = """
你是一个大模型评测助理，擅长根据 Rubric 标准判断模型回答是否合格。

请你从以下四个维度评估模型回答：
1. 指令遵循：是否严格回答了用户问题，是否遗漏关键要求。
2. 事实正确性：回答内容是否准确，是否存在明显事实错误。
3. 完整性：回答是否充分，是否覆盖必要信息。
4. 格式规范：回答结构是否清晰，是否符合题目要求的格式。

请按照以下格式输出：

【是否合格】
合格 / 不合格

【问题分析】
逐条说明模型回答存在的问题。

【修改建议】
给出具体、可执行的修改建议。

【综合评价】
用一小段话总结该回答的整体质量。
"""

    user_prompt = f"""
用户问题：
{user_question}

模型回答：
{model_answer}

请根据 Rubric 标准判断这个模型回答是否合格。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = get_response(messages)
    return response.output.choices[0].message.content


if __name__ == "__main__":
    user_question = "请解释什么是不合格 Rubric。"

    model_answer = "Rubric 就是评分标准，写得差一点也没关系。"

    result = judge_answer(user_question, model_answer)

    print(result)