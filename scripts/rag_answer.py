import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever import Retriever
from src.prompt_templates import RAG_PROMPT_TEMPLATE
from src.config import DEFAULT_TOP_K


def main():
    print("加载检索器...")
    try:
        retriever = Retriever()
    except Exception as e:
        print(f"错误: 无法加载索引，请先运行 build_index.py")
        print(f"详情: {e}")
        return
    
    while True:
        try:
            query = input("\n请输入你的问题 (q 退出): ").strip()
            
            if query.lower() == 'q':
                break
            
            if not query:
                continue
            
            print(f"\n正在检索相关标准片段...")
            results = retriever.search(query, DEFAULT_TOP_K)
            
            if not results:
                print("未找到相关结果")
                continue
            
            context_str = ""
            for i, res in enumerate(results, 1):
                context_str += f"\n[{i}] 来源: {res['source_file']} | 类型: {res['standard_type']} | 章节: {res['section']}\n"
                context_str += f"内容: {res['text']}\n"
            
            prompt = RAG_PROMPT_TEMPLATE.format(query=query, context=context_str)
            
            print(f"\n{'=' * 100}")
            print("生成的 RAG Prompt:")
            print(f"{'=' * 100}")
            print(prompt)
            print(f"{'=' * 100}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()
