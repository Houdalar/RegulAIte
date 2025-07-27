import os
from langchain_openai import ChatOpenAI

def get_llm():
    """
    Always return an OpenAI chat model for inference.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in your .env file to use OpenAI.")
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=api_key
    )