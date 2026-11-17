"""Small StateGraph with the slice of the LangGraph API these patterns use.

``add_node``, ``add_edge``, ``add_conditional_edges``, ``compile``, and
``invoke`` match how ``langgraph.graph.StateGraph`` is called in this repo.
When LangGraph is installed, the runners can compile the same builder against
the real library. The offline default is this graph, so the test suite does
not import LangGraph.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

START = "__start__"
END = "__end__"

Node = Callable[[dict], dict | None]
Route = Callable[[dict], Any]


class StateGraph:
    """Explicit state graph. Nodes return a partial state update."""

    def __init__(self, state_schema: Any = None) -> None:
        self.schema = state_schema
        self.nodes: dict[str, Node] = {}
        self.edges: dict[Any, Any] = {}
        self.conditionals: dict[Any, tuple[Route, dict]] = {}

    def add_node(self, name: str, fn: Node) -> StateGraph:
        if name in (START, END):
            raise ValueError(f"cannot add a node named {name!r}")
        if name in self.nodes:
            raise ValueError(f"node {name!r} already exists")
        self.nodes[name] = fn
        return self

    def add_edge(self, source: Any, dest: Any) -> StateGraph:
        if source in self.edges or source in self.conditionals:
            raise ValueError(f"{source!r} already has an outgoing edge")
        self.edges[source] = dest
        return self

    def add_conditional_edges(self, source: Any, path: Route, path_map: dict) -> StateGraph:
        if source in self.edges or source in self.conditionals:
            raise ValueError(f"{source!r} already has an outgoing edge")
        self.conditionals[source] = (path, dict(path_map))
        return self

    def compile(self) -> CompiledGraph:
        if START not in self.edges:
            raise ValueError("graph needs an edge from START")
        return CompiledGraph(self)


class CompiledGraph:
    """Runs a :class:`StateGraph` until it reaches END."""

    def __init__(self, graph: StateGraph) -> None:
        self.graph = graph

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        limit = 40
        if config and config.get("recursion_limit"):
            limit = int(config["recursion_limit"])
        current = self.graph.edges[START]
        merged = dict(state)
        steps = 0
        while current != END:
            steps += 1
            if steps > limit:
                raise RuntimeError(f"graph exceeded {limit} steps")
            if current not in self.graph.nodes:
                raise RuntimeError(f"unknown node {current!r}")
            update = self.graph.nodes[current](merged) or {}
            merged.update(update)
            current = self._next(current, merged)
        return merged

    def _next(self, node: Any, state: dict) -> Any:
        if node in self.graph.conditionals:
            path, path_map = self.graph.conditionals[node]
            key = path(state)
            if key in path_map:
                return path_map[key]
            return key
        if node not in self.graph.edges:
            raise RuntimeError(f"node {node!r} has no outgoing edge")
        return self.graph.edges[node]


def resolve_backend(backend: str) -> str:
    """``auto`` uses LangGraph when the package imports, otherwise the builtin graph."""
    if backend not in {"auto", "builtin", "langgraph"}:
        raise ValueError("backend must be 'auto', 'builtin', or 'langgraph'")
    if backend != "auto":
        return backend
    return "langgraph" if find_spec("langgraph") is not None else "builtin"


def graph_types(backend: str) -> tuple[Any, Any, Any, str]:
    """Return ``(StateGraph, START, END, runtime)`` for the requested backend."""
    runtime = resolve_backend(backend)
    if runtime == "langgraph":
        if find_spec("langgraph") is None:
            raise ImportError(
                "langgraph is not installed. Run: python -m pip install "
                "'jagadish-multi-agent-orchestration[frameworks]'"
            )
        from langgraph.graph import END as langgraph_end
        from langgraph.graph import START as langgraph_start
        from langgraph.graph import StateGraph as LangGraphStateGraph

        return LangGraphStateGraph, langgraph_start, langgraph_end, "langgraph"
    return StateGraph, START, END, "builtin"
