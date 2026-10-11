"""Tool registry with argument validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ToolError(Exception):
    """Base error for tool failures."""


class ToolNotFoundError(ToolError):
    """Raised when calling a tool that is not registered."""


class ToolArgumentError(ToolError):
    """Raised when tool arguments are missing or of the wrong type."""


@dataclass
class Tool:
    """A callable tool exposed to agents."""

    name: str
    description: str
    parameters: dict[str, type] = field(default_factory=dict)
    func: Callable[..., Any] = lambda **kwargs: None  # noqa: E731

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                key: getattr(tp, "__name__", str(tp))
                for key, tp in self.parameters.items()
            },
        }

    def validate(self, arguments: dict) -> None:
        for key, expected in self.parameters.items():
            if key not in arguments:
                raise ToolArgumentError(
                    f"tool {self.name!r} missing required argument {key!r}"
                )
            if not isinstance(arguments[key], expected):
                raise ToolArgumentError(
                    f"tool {self.name!r} argument {key!r} must be "
                    f"{expected.__name__}, got {type(arguments[key]).__name__}"
                )


class ToolRegistry:
    """Named registry of tools available to agents."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"unknown tool: {name!r}") from None

    def call(self, name: str, arguments: dict | None = None) -> Any:
        tool = self.get(name)
        args = dict(arguments or {})
        tool.validate(args)
        try:
            return tool.func(**args)
        except ToolError:
            raise
        except Exception as exc:  # wrap unexpected failures
            raise ToolError(f"tool {name!r} failed: {exc}") from exc

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
