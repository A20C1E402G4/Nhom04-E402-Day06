"""OpenAI GPT-4o provider implementation."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from vssa_agent.config import LLM_TEMPERATURE, OPENAI_API_KEY, OPENAI_MODEL

from .llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """Returns a `ChatOpenAI` configured from environment variables."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model or OPENAI_MODEL
        self._temperature = LLM_TEMPERATURE if temperature is None else temperature
        self._api_key = api_key or OPENAI_API_KEY

    def chat_model(self) -> BaseChatModel:
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            )
        return ChatOpenAI(
            model=self._model,
            temperature=self._temperature,
            api_key=self._api_key,
        )
