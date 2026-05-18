import sys
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, CHUNKS_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from src.document_loader import load_docx, load_xlsx, get_standard_type
from src.chunker import chunk_document_items


def main():
    files_processed = 0
    total_chunks = 0
    type_count = defaultdict(int)
    all_chunks = []
    
    docx_files = list(RAW_DATA_DIR.glob("*.docx"))
    xlsx_files = list(RAW_DATA_DIR.glob("*.xlsx"))
    all_files = docx_files + xlsx_files
    
    if not all_files:
        print(f"未在 {RAW_DATA_DIR} 找到任何 .docx 或 .xlsx 文件")
        return
    
    print(f"找到 {len(all_files)} 个文件，开始处理...")
    
    for file_path in tqdm(all_files):
        filename = file_path.name
        standard_type = get_standard_type(filename)
        
        if filename.endswith(".docx"):
            items = load_docx(file_path)
        elif filename.endswith(".xlsx"):
            items = load_xlsx(file_path)
        else:
            continue
        
        chunked_items = chunk_document_items(items, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunked_items):
            text = chunk.get("text", "").strip()
            if not text:
                continue
            
            chunk_id = f"{standard_type}_{str(total_chunks + 1).zfill(6)}"
            chunk_data = {
                "chunk_id": chunk_id,
                "text": text,
                "source_file": filename,
                "standard_type": standard_type,
                "section": chunk.get("section", ""),
                "content_type": chunk.get("content_type", "text"),
                "page_or_sheet": chunk.get("sheet", None),
                "char_count": chunk.get("char_count", len(text))
            }
            all_chunks.append(chunk_data)
            type_count[standard_type] += 1
            total_chunks += 1
        
        files_processed += 1
    
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"\n处理完成！")
    print(f"处理文件数: {files_processed}")
    print(f"生成 chunks 数: {total_chunks}")
    print(f"\n各类型分布:")
    for stype, count in sorted(type_count.items()):
        print(f"  {stype}: {count}")
    print(f"\n输出文件: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
