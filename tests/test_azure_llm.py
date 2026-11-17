"""Azure OpenAI client: env switch, request shape, and error mapping. No network."""

import io
import json
import urllib.error

import pytest

from orchestrator.core import Message, MessageRole
from orchestrator.llm import (
    AzureOpenAIClient,
    AzureOpenAIError,
    FakeLLM,
    azure_config,
    azure_configured,
    build_llm,
)
from orchestrator.llm.client import ToolCall


def test_azure_is_unconfigured_by_default():
    assert azure_config() is None
    assert azure_configured() is False
    assert isinstance(build_llm(), FakeLLM)


def test_azure_needs_all_three_settings(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    assert azure_configured() is False
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-test")
    config = azure_config()
    assert config is not None
    assert config["deployment"] == "gpt-test"
    assert config["api_version"] == "2024-10-21"


def test_build_llm_returns_azure_client_when_configured(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-test")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    client = build_llm()
    assert isinstance(client, AzureOpenAIClient)
    assert client.deployment == "gpt-test"
    assert client.api_version == "2024-06-01"
    assert "api-key" not in client._url()


def test_complete_posts_chat_completions_and_maps_tools(monkeypatch):
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "model": "gpt-test",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-7",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"query": "capital of France"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                }
            ).encode()

    def _urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    client = AzureOpenAIClient(
        endpoint="https://example.openai.azure.com",
        api_key="secret-key",
        deployment="gpt test",
        api_version="2024-10-21",
    )
    response = client.complete(
        [Message(MessageRole.USER, "Where is the capital?")],
        tools=[{"name": "search", "description": "lookup", "parameters": {"query": "str"}}],
    )
    assert "/openai/deployments/gpt%20test/chat/completions" in captured["url"]
    assert "api-version=2024-10-21" in captured["url"]
    assert captured["headers"]["Api-key"] == "secret-key"
    assert captured["body"]["messages"] == [{"role": "user", "content": "Where is the capital?"}]
    assert captured["body"]["tools"][0]["function"]["name"] == "search"
    assert response.content == ""
    assert response.tool_calls == [
        ToolCall(name="search", arguments={"query": "capital of France"}, call_id="call-7")
    ]
    assert response.usage["source"] == "azure"
    assert response.usage["total_tokens"] == 15
    assert "secret-key" not in captured["url"]


def test_http_error_does_not_include_the_api_key(monkeypatch):
    def _urlopen(request, timeout=0):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error": {"message": "bad credentials"}}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    client = AzureOpenAIClient("https://example.openai.azure.com", "super-secret", "gpt")
    with pytest.raises(AzureOpenAIError, match="bad credentials") as caught:
        client.complete([Message(MessageRole.USER, "hi")])
    assert "super-secret" not in str(caught.value)


def test_http_endpoint_and_boolean_schema_are_rejected_or_mapped():
    with pytest.raises(ValueError, match="https"):
        AzureOpenAIClient("http://example.openai.azure.com", "k", "gpt")
    from orchestrator.llm.azure import _tool_payload

    payload = _tool_payload(
        {"name": "flag", "description": "set it", "parameters": {"enabled": "bool", "count": "int"}}
    )
    properties = payload["function"]["parameters"]["properties"]
    assert properties["enabled"]["type"] == "boolean"
    assert properties["count"]["type"] == "number"


def test_missing_choices_raise(monkeypatch):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices": []}'

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=0: _Response())
    client = AzureOpenAIClient("https://example.openai.azure.com", "k", "gpt")
    with pytest.raises(AzureOpenAIError, match="no choices"):
        client.complete([Message(MessageRole.USER, "hi")])
