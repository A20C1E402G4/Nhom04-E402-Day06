"""Abstract base class for LLM providers used by the VSSA agent."""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class LLMProvider(ABC):
    """Common interface so we can swap GPT-4o ↔ Gemini ↔ local models."""

    @abstractmethod
    def chat_model(self) -> BaseChatModel:
        """Return a LangChain chat model ready to be `bind_tools`-ed."""
