"""Offline built-in tools: calculator, search stub, and clock.

All built-ins are deterministic and require no network access, so they are
safe to use inside the offline test suite.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from .registry import Tool, ToolError

_ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        try:
            return _ARITHMETIC[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ToolError("division by zero") from None
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITHMETIC:
        return _ARITHMETIC[type(node.op)](_eval_node(node.operand))
    raise ToolError(f"unsupported expression: {ast.dump(node)}")


def _calculate(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression {expression!r}: {exc}") from None
    return _eval_node(tree)


def make_calculator_tool() -> Tool:
    """Safe arithmetic evaluator (no ``eval`` of arbitrary code)."""

    def _run(expression: str) -> float:
        return _calculate(expression)

    return Tool(
        name="calculator",
        description="Evaluate a basic arithmetic expression, e.g. '2*(3+4)'.",
        parameters={"expression": str},
        func=_run,
    )


_DEFAULT_CORPUS = {
    "login error": "Known issue KB-1042: login errors after password reset; clear cookies and retry.",
    "password reset": "Docs: password reset links expire after 30 minutes.",
    "refund policy": "Refunds are issued to the original payment method within 5-7 days.",
}


def make_search_stub_tool(corpus: dict[str, str] | None = None) -> Tool:
    """Deterministic stub for a search/retrieval backend."""

    data = dict(_DEFAULT_CORPUS if corpus is None else corpus)

    def _run(query: str) -> str:
        lowered = query.lower()
        for key, value in data.items():
            if key.lower() in lowered:
                return value
        return f"No results for {query!r} (stub)."

    return Tool(
        name="search",
        description="Search the stub knowledge base for a query.",
        parameters={"query": str},
        func=_run,
    )


def make_clock_tool(fixed_iso: str = "2026-01-01T00:00:00+00:00") -> Tool:
    """Deterministic clock returning a fixed timestamp (test-friendly)."""

    def _run() -> str:
        return fixed_iso

    return Tool(
        name="clock",
        description="Return the current time as an ISO-8601 string (fixed stub).",
        parameters={},
        func=_run,
    )


def default_registry() -> Any:
    """Registry pre-loaded with the three built-in tools."""
    from .registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(make_calculator_tool())
    registry.register(make_search_stub_tool())
    registry.register(make_clock_tool())
    return registry
