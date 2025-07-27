import os

from langchain_openai import ChatOpenAI

from .llm_transformers import TransformersLLM


def get_llm(provider: str = "openai"):
    """Return an LLM instance based on the provider."""
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in your .env file to use OpenAI.")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)
    if provider == "qwen":
        return TransformersLLM(
            model_name="Qwen/Qwen3-235B-A22B-Instruct-2507",
            device_map="auto",
        )
    raise ValueError(f"Unknown provider: {provider}")
