"""Azure OpenAI settings read from the environment.

This module imports neither the Azure client nor the factory, so those two
can depend on it without importing each other.
"""

from __future__ import annotations

import os

DEFAULT_API_VERSION = "2024-10-21"

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
