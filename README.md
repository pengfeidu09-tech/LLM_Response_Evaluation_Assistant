# Rubric 标准知识库 RAG 助手

面向大模型文本标注与 Rubric 标注场景的标准知识库 RAG 助手。

## 项目背景

本项目旨在构建一个本地知识库，用于存储和检索各类文本标注标准（如 Rubric 标注标准、题目质量标注规范、文生文评测标准等），支持快速查找相关标准片段。

## 技术栈

- **向量模型**: BGE-M3
- **向量数据库**: FAISS
- **前端**: Streamlit
- **文档处理**: python-docx, pandas, openpyxl

## 目录结构

```
rubric_rag_assistant/
├── data/
│   ├── raw/              # 原始文档（.docx, .xlsx）
│   ├── processed/        # 处理后的 chunks
│   └── metadata/         # 元数据
├── models/
│   └── bge-m3/           # BGE-M3 模型文件
├── vector_store/         # FAISS 向量索引
├── scripts/              # 脚本文件
│   ├── extract_documents.py
│   ├── build_index.py
│   ├── search_test.py
│   └── rag_answer.py
├── src/                  # 源代码模块
│   ├── config.py
│   ├── document_loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── prompt_templates.py
├── app.py                # Streamlit 应用
├── requirements.txt
└── README.md
```

## 使用说明

### 1. 放置原始文件

将需要处理的标注标准文件（.docx 或 .xlsx）放入 `data/raw/` 目录。

### 2. 提取文档并生成 chunks

```bash
python scripts/extract_documents.py
```

该脚本会：
- 遍历 data/raw/ 目录下的 .docx 和 .xlsx 文件
- 提取内容并切分成 chunks（chunk_size=800, overlap=120）
- 自动识别标准类型
- 输出到 data/processed/chunks.jsonl

### 3. 构建向量索引

```bash
python scripts/build_index.py
```

该脚本会：
- 加载 chunks.jsonl
- 使用 BGE-M3 生成 embedding
- 构建 FAISS IndexFlatIP 索引
- 保存索引和元数据

### 4. 测试检索

```bash
python scripts/search_test.py
```

提供交互式命令行测试，包含预设的测试问题。

### 5. 生成 RAG Prompt

```bash
python scripts/rag_answer.py
```

输入问题，自动检索相关片段并生成可复制给大模型的 RAG Prompt。

### 6. 启动 Streamlit 网页界面

```bash
streamlit run app.py
```

访问 http://localhost:8501 使用网页界面。

## 标准类型识别规则

- 文件名包含 "Rubric" 或 "Rubrics" → rubric_quality
- 文件名包含 "题目质量" → prompt_quality
- 文件名包含 "文生文" 或 "评测标准" → answer_evaluation
- 文件名包含 "字数" → word_count_rule
- 文件名包含 "标签体系" → label_taxonomy
- 文件名包含 "人设" → persona_info
- 其他 → general_standard

## 后续升级方向

- [ ] BM25 + FAISS 混合检索
- [ ] 相似 chunk 去重
- [ ] Rubric 质量检查 Agent
- [ ] 标注建议 Agent
- [ ] LoRA 微调分类模型
- [ ] vLLM 部署
