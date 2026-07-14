"""
Single point of contact with the local model.

Everything downstream (chat endpoint, and later the RAG/LangGraph/Guardrails
layers) talks to a LangChain BaseChatModel, not to Ollama directly. Swapping
the local model for a hosted one later means changing this one function.
"""
from langchain_ollama import ChatOllama

from app.config import settings


def get_chat_model() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
    )
