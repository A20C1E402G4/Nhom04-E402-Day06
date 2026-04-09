"""Abstract base class for LLM providers used by the VSSA agent."""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class LLMProvider(ABC):
    """
    Abstract Base Class for LLM Providers.
    Supports OpenAI, Gemini, and Local models.
    """

    @abstractmethod
    def chat_model(self) -> BaseChatModel:
        """Return a LangChain chat model instance ready for use."""
        ...
