#!/usr/bin/env python
# coding: utf-8

import os
import json
from pathlib import Path

import dashscope
import streamlit as st
from dotenv import load_dotenv


# ========== 1. 读取 .env ==========

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("DASHSCOPE_API_KEY")
model_name = os.getenv("MODEL_NAME", "deepseek-r1")

if not api_key:
    st.error("没有读取到 DASHSCOPE_API_KEY，请检查 .env 文件。")
    st.stop()

dashscope.api_key = api_key


# ========== 2. System Prompt ==========

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


# ========== 3. 工具函数 ==========

def extract_json(text):
    """
    从模型输出中提取 JSON。
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


# ========== 4. Streamlit 页面 ==========

st.set_page_config(
    page_title="LLM Response Evaluation Assistant",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 基于 Rubric 的模型回答质量评测助手")
st.caption("LLM Response Evaluation Assistant v0.5")

st.markdown("""
这个工具用于辅助判断模型回答是否满足 Rubric 标准。  
输入用户题目、模型回答和 Rubric 后，系统会自动输出结构化评测结果。
""")

with st.sidebar:
    st.header("当前配置")
    st.write("模型：", model_name)
    st.write("API Key：", "已读取" if api_key else "未读取")

    st.markdown("---")
    st.markdown("### 使用步骤")
    st.markdown("""
1. 输入用户题目  
2. 输入模型回答  
3. 输入 Rubric 标准  
4. 点击开始评测  
5. 查看结果  
""")


question = st.text_area(
    "用户题目",
    height=160,
    placeholder="请在这里粘贴用户题目..."
)

answer = st.text_area(
    "模型回答",
    height=220,
    placeholder="请在这里粘贴模型回答..."
)

rubrics = st.text_area(
    "Rubric 标准",
    height=260,
    placeholder="请在这里粘贴 Rubric 标准，可以包含多条..."
)

submit = st.button("开始评测", type="primary")

if submit:
    if not question.strip():
        st.warning("请先输入用户题目。")
    elif not answer.strip():
        st.warning("请先输入模型回答。")
    elif not rubrics.strip():
        st.warning("请先输入 Rubric 标准。")
    else:
        with st.spinner("正在调用模型评测，请稍等..."):
            raw_output = call_model(question, answer, rubrics)

        st.subheader("模型原始输出")
        st.text_area("Raw Output", value=raw_output, height=260)

        st.subheader("结构化评测结果")

        try:
            parsed_result = extract_json(raw_output)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("总体结论", parsed_result.get("overall_result", ""))
            with col2:
                st.metric("评分", parsed_result.get("score", ""))

            st.markdown("### 维度分析")
            st.json(parsed_result.get("dimension_analysis", {}))

            st.markdown("### 逐条 Rubric 判断")
            st.json(parsed_result.get("rubric_check", []))

            st.markdown("### 主要问题")
            for item in parsed_result.get("main_problems", []):
                st.write("- " + item)

            st.markdown("### 修改建议")
            for item in parsed_result.get("suggestions", []):
                st.write("- " + item)

            st.markdown("### 最终评价")
            st.info(parsed_result.get("final_comment", ""))

            json_str = json.dumps(parsed_result, ensure_ascii=False, indent=2)

            st.download_button(
                label="下载 JSON 结果",
                data=json_str,
                file_name="evaluation_result.json",
                mime="application/json"
            )

        except Exception as e:
            st.error("JSON 解析失败，模型可能没有严格输出 JSON。")
            st.write("错误信息：", e)
