"""Tool registry, tool-calling agent loop, and offline built-in tools."""

from .builtins import make_calculator_tool, make_clock_tool, make_search_stub_tool
from .loop import MaxIterationsError, ToolCallingLoop
from .registry import Tool, ToolArgumentError, ToolError, ToolNotFoundError, ToolRegistry

__all__ = [
    "MaxIterationsError",
    "Tool",
    "ToolArgumentError",
    "ToolCallingLoop",
    "ToolError",
    "ToolNotFoundError",
    "ToolRegistry",
    "make_calculator_tool",
    "make_clock_tool",
    "make_search_stub_tool",
]
