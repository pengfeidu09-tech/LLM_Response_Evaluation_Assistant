# LLM Response Evaluation Assistant

## 项目简介

本项目是一个基于 Rubric 的大模型回答质量评测助手，面向大模型标注、Rubric 评测和模型回答质量分析场景。

用户可以在网页端输入原始题目、模型回答和 Rubric 标准，系统会调用大模型接口，从指令遵循、事实正确性、完整性、格式规范等维度生成结构化评测结果，并输出总体结论、评分、逐条 Rubric 判断、问题分析和修改建议。

## 项目背景

本人目前参与大模型文本标准与 Rubric 标注相关工作。在实际标注过程中，人工逐条判断模型回答是否符合 Rubric 标准耗时较长，且不同标注员之间可能存在理解不一致的问题。

因此，本项目尝试搭建一个 AI 辅助评测工具，用于提升模型回答质量判断和人工复核效率。

## 当前功能

- 支持从 `.env` 文件读取 API Key 和模型配置
- 支持调用 DashScope / deepseek-r1 模型
- 支持网页端输入用户题目、模型回答和 Rubric 标准
- 支持输出 JSON 结构化评测结果
- 支持展示总体结论、评分、逐条 Rubric 判断、问题分析和修改建议
- 支持下载 JSON 结果
- 支持命令行单条评测
- 支持 Excel 批量评测脚本

## 技术栈

- Python
- Streamlit
- DashScope SDK
- deepseek-r1
- python-dotenv
- pandas
- openpyxl
- Prompt Engineering

## 项目结构

```text
LLM_Response_Evaluation_Assistant/
├── app.py
├── rubric_judge.py
├── rubric_judge_v03_json.py
├── batch_rubric_judge_v04.py
├── requirements.txt
├── README.md
└── .env.example