"""Choose the offline mock or Azure OpenAI from the environment.

The default is always the deterministic fake. Azure is selected only when all
three of endpoint, api key, and deployment are present, so a half-configured
shell still runs offline.
"""

from __future__ import annotations

import os

from .azure import DEFAULT_API_VERSION, AzureOpenAIClient
from .client import LLMClient
from .fake import FakeLLM

ENV_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
ENV_API_KEY = "AZURE_OPENAI_API_KEY"
ENV_DEPLOYMENT = "AZURE_OPENAI_DEPLOYMENT"
ENV_DEPLOYMENT_ALIAS = "AZURE_OPENAI_CHAT_DEPLOYMENT"
ENV_API_VERSION = "AZURE_OPENAI_API_VERSION"


def azure_config() -> dict[str, str] | None:
    """Return Azure settings, or ``None`` when the env is not fully set."""
    endpoint = os.environ.get(ENV_ENDPOINT, "").strip()
    api_key = os.environ.get(ENV_API_KEY, "").strip()
    deployment = os.environ.get(ENV_DEPLOYMENT, "").strip() or os.environ.get(
        ENV_DEPLOYMENT_ALIAS, ""
    ).strip()
    if not (endpoint and api_key and deployment):
        return None
    api_version = os.environ.get(ENV_API_VERSION, "").strip() or DEFAULT_API_VERSION
    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "deployment": deployment,
        "api_version": api_version,
    }


def azure_configured() -> bool:
    return azure_config() is not None


def build_llm(fake: LLMClient | None = None) -> LLMClient:
    """Return an Azure client when configured, otherwise ``fake`` or :class:`FakeLLM`."""
    config = azure_config()
    if config is not None:
        return AzureOpenAIClient(**config)
    if fake is not None:
        return fake
    return FakeLLM()
