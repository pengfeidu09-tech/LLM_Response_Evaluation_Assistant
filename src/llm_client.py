import os
from pathlib import Path
from dotenv import load_dotenv
import dashscope


def load_llm_config():
    """
    从项目根目录的 .env 文件中读取配置。
    """
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    api_key = os.getenv("DASHSCOPE_API_KEY")
    model_name = os.getenv("MODEL_NAME", "deepseek-r1")

    if not api_key:
        raise ValueError("没有读取到 DASHSCOPE_API_KEY，请检查项目根目录下的 .env 文件。")

    return api_key, model_name


def call_dashscope(messages):
    """
    使用 DashScope 调用大模型。
    返回 response.output.choices[0].message.content
    """
    api_key, model_name = load_llm_config()
    dashscope.api_key = api_key

    response = dashscope.Generation.call(
        model=model_name,
        messages=messages,
        result_format="message"
    )

    return response.output.choices[0].message.content
