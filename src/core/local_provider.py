"""Local Ollama provider implementation."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from .llm_provider import LLMProvider


class LocalProvider(LLMProvider):
    """Returns a `ChatOllama` configured for a local Ollama model."""

    def __init__(
        self,
        model: str = "qwen2.5",
        temperature: float = 0,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._base_url = base_url

    def chat_model(self) -> BaseChatModel:
        return ChatOllama(
            model=self._model,
            temperature=self._temperature,
            base_url=self._base_url,
        )
