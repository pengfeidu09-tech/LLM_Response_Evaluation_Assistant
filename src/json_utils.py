import json


def extract_json(text):
    """
    从模型输出中提取 JSON。
    支持去掉 ```json 和 ```。
    自动截取第一个 { 到最后一个 }。
    解析失败时抛出异常，不静默吞掉。
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
