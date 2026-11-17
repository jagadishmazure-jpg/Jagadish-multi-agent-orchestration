"""Azure OpenAI chat-completions client.

Used only when ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``, and
``AZURE_OPENAI_DEPLOYMENT`` are set. The offline suite never constructs it
unless a test opts in and stubs the HTTP call.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote

from ..core.messages import Message, MessageRole
from .client import LLMResponse, ToolCall
from .config import DEFAULT_API_VERSION, azure_config


class AzureOpenAIError(RuntimeError):
    """The Azure OpenAI request failed or returned an unusable payload."""


class AzureOpenAIClient:
    """Minimal chat-completions client for one Azure OpenAI deployment.

    Talks to the REST API with the standard library so the default install
    stays free of the ``openai`` package. Tool calls are mapped into
    :class:`LLMResponse` so the existing tool loop can use the client.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = DEFAULT_API_VERSION,
        timeout: float = 60.0,
    ) -> None:
        if not endpoint or not api_key or not deployment:
            raise ValueError("endpoint, api_key, and deployment are required")
        self.endpoint = endpoint.strip().rstrip("/")
        self.api_key = api_key
        self.deployment = deployment.strip()
        self.api_version = api_version.strip() or DEFAULT_API_VERSION
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> AzureOpenAIClient:
        config = azure_config()
        if config is None:
            raise AzureOpenAIError(
                "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT."
            )
        return cls(**config)

    def _url(self) -> str:
        root = self.endpoint
        if root.endswith("/openai"):
            root = root[: -len("/openai")]
        deployment = quote(self.deployment, safe="")
        version = quote(self.api_version, safe="")
        return (
            f"{root}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={version}"
        )

    def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "messages": [_message_payload(message) for message in messages],
        }
        if tools:
            body["tools"] = [_tool_payload(tool) for tool in tools]
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self._url(),
            data=data,
            headers={
                "Content-Type": "application/json",
                "api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _http_error_detail(exc)
            raise AzureOpenAIError(f"Azure OpenAI HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise AzureOpenAIError(f"Azure OpenAI request failed: {exc.reason}") from None
        except json.JSONDecodeError as exc:
            raise AzureOpenAIError(f"Azure OpenAI returned invalid JSON: {exc}") from None
        return _parse_response(payload)


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    calls = message.metadata.get("tool_calls") or []
    if message.role == MessageRole.ASSISTANT and calls:
        payload["tool_calls"] = [
            {
                "id": call.get("call_id") or f"call-{index}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments") or {}, sort_keys=True),
                },
            }
            for index, call in enumerate(calls, start=1)
        ]
    if message.role == MessageRole.TOOL:
        payload["tool_call_id"] = str(message.metadata.get("call_id") or "call")
    return payload


def _tool_payload(schema: dict) -> dict:
    if schema.get("type") == "function":
        return schema
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []
    for name, type_name in (schema.get("parameters") or {}).items():
        json_type = "number" if type_name in {"int", "float"} else "string"
        properties[name] = {"type": json_type, "description": name}
        required.append(name)
    return {
        "type": "function",
        "function": {
            "name": schema.get("name", ""),
            "description": schema.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _parse_response(payload: dict) -> LLMResponse:
    try:
        choice = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AzureOpenAIError("Azure OpenAI response had no choices[0].message") from exc
    content = choice.get("content") or ""
    tool_calls: list[ToolCall] = []
    for index, call in enumerate(choice.get("tool_calls") or [], start=1):
        function = call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError as exc:
                raise AzureOpenAIError(f"tool arguments were not JSON: {exc}") from None
        else:
            parsed = dict(arguments)
        tool_calls.append(
            ToolCall(
                name=function.get("name", ""),
                arguments=parsed,
                call_id=str(call.get("id") or f"call-{index}"),
            )
        )
    usage = payload.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        model=str(payload.get("model") or "azure-openai"),
        usage={
            "source": "azure",
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": int(usage.get("total_tokens") or (prompt + completion)),
        },
    )


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return exc.reason if isinstance(exc.reason, str) else "request rejected"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:300] or "request rejected"
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])[:300]
    return raw[:300] or "request rejected"
