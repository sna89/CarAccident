import os
from langchain_openai import ChatOpenAI


def get_llm_model(provider, model_name):
    if provider == "openai":
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        llm = ChatOpenAI(model=model_name, openai_api_key=openai_api_key, temperature=0)
        return llm

    else:
        return None