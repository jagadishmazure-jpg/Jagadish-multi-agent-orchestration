"""Choose the offline mock or Azure OpenAI from the environment.

The default is always the deterministic fake. Azure is selected only when all
three of endpoint, api key, and deployment are present, so a half-configured
shell still runs offline.
"""

from __future__ import annotations

from .azure import AzureOpenAIClient
from .client import LLMClient
from .config import azure_config, azure_configured
from .fake import FakeLLM

__all__ = ["azure_config", "azure_configured", "build_llm"]


def build_llm(fake: LLMClient | None = None) -> LLMClient:
    """Return an Azure client when configured, otherwise ``fake`` or :class:`FakeLLM`."""
    config = azure_config()
    if config is not None:
        return AzureOpenAIClient(**config)
    if fake is not None:
        return fake
    return FakeLLM()
