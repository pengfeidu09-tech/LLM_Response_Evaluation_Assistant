# -*- coding: utf-8 -*-

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_retriever_tf import retrieve_standards


def print_result(result, rank):
    print(f"\n{'=' * 100}")
    print(f"Rank: {rank}")
    print(f"Score: {round(result['score'], 4)}")
    print(f"standard_type: {result.get('standard_type', '')}")
    print(f"source_file: {result.get('source_file', '')}")
    print(f"section: {result.get('section', '')}")
    print(f"\n{result.get('text', '')[:300]}")
    print(f"{'=' * 100}")


def main():
    test_questions = [
        "什么是不合格 Rubric？",
        "题目无效和题目模糊有什么区别？",
        "强时效和弱时效怎么区分？"
    ]

    print("加载检索器...")
    print()

    for question in test_questions:
        print(f"\n{'#' * 100}")
        print(f"【测试问题】：{question}")
        print(f"{'#' * 100}")

        try:
            results = retrieve_standards(question, top_k=5)

            if not results:
                print("未找到相关结果")
                continue

            for i, res in enumerate(results, 1):
                print_result(res, i)

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 100)
    print("测试完成")
    print("=" * 100)


if __name__ == "__main__":
    main()
