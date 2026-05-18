# -*- coding: utf-8 -*-

import os
import sys
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_retriever_tf import retrieve_standards
from src.judge_prompts import JUDGE_SYSTEM_PROMPT
from src.llm_client import call_dashscope, load_llm_config
from src.json_utils import extract_json


st.set_page_config(
    page_title="基于 RAG 的 Rubric 标注与回答质量评测助手",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 基于 RAG 的 Rubric 标注与回答质量评测助手")
st.markdown("""
本工具用于辅助大模型文本标注、Rubric 标注和模型回答质量评测。系统会先从本地标注标准知识库中检索相关标准片段，再调用大模型输出结构化评测结果。
""")


# ========== 侧边栏 ==========

PROJECT_ROOT = Path(__file__).parent

with st.sidebar:
    st.header("系统状态")

    # 读取环境变量
    try:
        api_key, model_name = load_llm_config()
        st.success(f"✅ MODEL_NAME: {model_name}")
        st.success("✅ API Key: 已读取")
    except Exception as e:
        st.error(f"❌ API Key: {e}")
        model_name = "未配置"

    # 检查文件是否存在
    index_exists = (PROJECT_ROOT / "vector_store" / "rubric_kb.faiss").exists()
    metadata_exists = (PROJECT_ROOT / "data" / "metadata" / "metadata.json").exists()
    model_exists = (PROJECT_ROOT / "models" / "bge-m3").exists()

    if index_exists:
        st.success("✅ FAISS 索引: 存在")
    else:
        st.error("❌ FAISS 索引: 不存在")

    if metadata_exists:
        st.success("✅ metadata: 存在")
    else:
        st.error("❌ metadata: 不存在")

    if model_exists:
        st.success("✅ BGE-M3 模型: 存在")
    else:
        st.error("❌ BGE-M3 模型: 不存在")

    st.markdown("---")
    st.header("说明")
    st.markdown("""
- 先点击「检索相关标准」查看检索结果
- 再点击「开始 RAG 评测」进行完整评测
- 结果支持下载为 JSON 文件
    """)


# ========== 主界面输入框 ==========

question = st.text_area("用户题目", placeholder="请输入用户题目...", height=150)
answer = st.text_area("模型回答", placeholder="请输入模型回答...", height=200)
rubrics = st.text_area("Rubric 标准", placeholder="请输入 Rubric 标准...", height=250)


# ========== 按钮 ==========

col1, col2 = st.columns(2)

with col1:
    btn_retrieve = st.button("检索相关标准", type="secondary", use_container_width=True)

with col2:
    btn_judge = st.button("开始 RAG 评测", type="primary", use_container_width=True)


# ========== 检索结果展示 ==========

if "retrieved_items" not in st.session_state:
    st.session_state.retrieved_items = None

if "raw_output" not in st.session_state:
    st.session_state.raw_output = None

if "parsed_result" not in st.session_state:
    st.session_state.parsed_result = None


def show_retrieved_items(items):
    st.subheader("📚 检索到的标准片段")
    for i, item in enumerate(items, start=1):
        with st.expander(f"Rank {i} - Score: {round(item['score'], 4)} - {item.get('section', 'No Section')}", expanded=(i == 1)):
            st.markdown(f"**Score**: {round(item['score'], 4)}")
            st.markdown(f"**standard_type**: {item.get('standard_type', '')}")
            st.markdown(f"**source_file**: {item.get('source_file', '')}")
            st.markdown(f"**section**: {item.get('section', '')}")
            st.markdown("**text**:")
            st.markdown(f"> {item.get('text', '')}")


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


# ========== 检索按钮逻辑 ==========

if btn_retrieve:
    if not question or not rubrics:
        st.warning("请先输入用户题目和 Rubric 标准")
    else:
        with st.spinner("正在检索相关标准..."):
            try:
                retrieval_query = question + "\n" + rubrics
                retrieved_items = retrieve_standards(retrieval_query, top_k=5)
                st.session_state.retrieved_items = retrieved_items
                st.session_state.raw_output = None
                st.session_state.parsed_result = None

                show_retrieved_items(retrieved_items)

            except Exception as e:
                st.error(f"检索失败: {e}")
                import traceback
                st.exception(e)


# ========== 评测按钮逻辑 ==========

if btn_judge:
    if not question or not answer or not rubrics:
        st.warning("请输入用户题目、模型回答和 Rubric 标准")
    else:
        with st.spinner("正在进行 RAG 评测..."):
            try:
                # 第一步：检索
                retrieval_query = question + "\n" + rubrics
                retrieved_items = retrieve_standards(retrieval_query, top_k=5)
                st.session_state.retrieved_items = retrieved_items

                # 第二步：构造 messages
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

                # 第三步：调用 LLM
                raw_output = call_dashscope(messages)
                st.session_state.raw_output = raw_output

                # 第四步：解析 JSON
                try:
                    parsed_result = extract_json(raw_output)
                    st.session_state.parsed_result = parsed_result
                except Exception as e:
                    st.warning(f"JSON 解析失败: {e}")
                    parsed_result = None

                # 展示结果
                show_retrieved_items(retrieved_items)

                st.subheader("💬 模型原始输出")
                st.text_area("原始输出", value=raw_output, height=300)

                if parsed_result:
                    st.subheader("📊 结构化评测结果")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("总体结论", parsed_result.get("overall_result", ""))
                    with col2:
                        st.metric("评分", parsed_result.get("score", ""))

                    st.markdown("### 维度分析")
                    st.json(parsed_result.get("dimension_analysis", {}))

                    st.markdown("### Rubric 检查")
                    st.json(parsed_result.get("rubric_check", []))

                    st.markdown("### 检索到的标准依据")
                    st.json(parsed_result.get("retrieved_standard_basis", []))

                    st.markdown("### 主要问题")
                    main_problems = parsed_result.get("main_problems", [])
                    if main_problems:
                        for problem in main_problems:
                            st.markdown(f"- {problem}")
                    else:
                        st.markdown("无")

                    st.markdown("### 修改建议")
                    suggestions = parsed_result.get("suggestions", [])
                    if suggestions:
                        for suggestion in suggestions:
                            st.markdown(f"- {suggestion}")
                    else:
                        st.markdown("无")

                    st.markdown("### 最终评价")
                    st.info(parsed_result.get("final_comment", ""))

                    # 下载按钮
                    json_str = json.dumps(parsed_result, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="下载 JSON 结果",
                        data=json_str,
                        file_name="judgment_result.json",
                        mime="application/json"
                    )
                else:
                    st.warning("未能解析为结构化 JSON，请参考模型原始输出")

            except Exception as e:
                st.error(f"评测失败: {e}")
                import traceback
                st.exception(e)


# ========== 展示已有的 session state 结果 ==========

elif st.session_state.retrieved_items:
    show_retrieved_items(st.session_state.retrieved_items)

    if st.session_state.raw_output:
        st.subheader("💬 模型原始输出")
        st.text_area("原始输出", value=st.session_state.raw_output, height=300)

        if st.session_state.parsed_result:
            parsed_result = st.session_state.parsed_result

            st.subheader("📊 结构化评测结果")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("总体结论", parsed_result.get("overall_result", ""))
            with col2:
                st.metric("评分", parsed_result.get("score", ""))

            st.markdown("### 维度分析")
            st.json(parsed_result.get("dimension_analysis", {}))

            st.markdown("### Rubric 检查")
            st.json(parsed_result.get("rubric_check", []))

            st.markdown("### 检索到的标准依据")
            st.json(parsed_result.get("retrieved_standard_basis", []))

            st.markdown("### 主要问题")
            main_problems = parsed_result.get("main_problems", [])
            if main_problems:
                for problem in main_problems:
                    st.markdown(f"- {problem}")
            else:
                st.markdown("无")

            st.markdown("### 修改建议")
            suggestions = parsed_result.get("suggestions", [])
            if suggestions:
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
            else:
                st.markdown("无")

            st.markdown("### 最终评价")
            st.info(parsed_result.get("final_comment", ""))

            # 下载按钮
            json_str = json.dumps(parsed_result, ensure_ascii=False, indent=2)
            st.download_button(
                label="下载 JSON 结果",
                data=json_str,
                file_name="judgment_result.json",
                mime="application/json"
            )
