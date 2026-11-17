"""LLM client interface, deterministic fakes, and the optional Azure client."""

from .azure import AzureOpenAIClient, AzureOpenAIError
from .client import LLMClient, LLMResponse, ToolCall
from .factory import azure_config, azure_configured, build_llm
from .fake import FakeLLM
from .meter import MOCK_CALL_MS, MeteredLLM

__all__ = [
    "MOCK_CALL_MS",
    "AzureOpenAIClient",
    "AzureOpenAIError",
    "FakeLLM",
    "LLMClient",
    "LLMResponse",
    "MeteredLLM",
    "ToolCall",
    "azure_config",
    "azure_configured",
    "build_llm",
]
