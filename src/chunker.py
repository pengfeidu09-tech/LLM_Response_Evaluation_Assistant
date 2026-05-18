from copy import deepcopy


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120):
    """
    将长文本切分为多个 chunk。
    安全点：
    1. end 到达文本末尾后立刻 break，避免最后一段无限重复。
    2. 保证 start 每轮都会前进。
    3. 避免 chunk_overlap >= chunk_size 导致死循环。
    """
    if text is None:
        return []

    text = str(text).strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        chunk_overlap = 0

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    text_len = len(text)

    if text_len <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = end - chunk_overlap

        if next_start <= start:
            next_start = start + chunk_size

        start = next_start

    return chunks


def chunk_document_items(items, chunk_size: int = 800, chunk_overlap: int = 120):
    """
    对 document_loader 输出的 items 进行切分。
    每个 item 应至少包含 text 字段。
    切分后保留原 metadata。
    """
    chunked_items = []
    global_idx = 0

    for item in items:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        if not text:
            continue

        pieces = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for local_idx, piece in enumerate(pieces):
            new_item = deepcopy(item)
            new_item["text"] = piece
            
            # 补充默认字段
            new_item["content_type"] = new_item.get("content_type", "text")
            new_item["source_file"] = new_item.get("source_file", "")
            new_item["standard_type"] = new_item.get("standard_type", "general_standard")
            new_item["section"] = new_item.get("section", "")
            new_item["page_or_sheet"] = new_item.get("page_or_sheet", "")
            new_item["char_count"] = len(piece)

            old_chunk_id = str(item.get("chunk_id", "chunk"))
            new_item["chunk_id"] = f"{old_chunk_id}_{local_idx:04d}_{global_idx:06d}"

            chunked_items.append(new_item)
            global_idx += 1

    return chunked_items
