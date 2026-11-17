"""In-process A2A mesh.

Nothing in this repository registered agents before the research crew. The
mesh is a name-to-server map: discover returns agent cards, send delivers
``message/send``. An external mesh that can fetch an agent card and POST
JSON-RPC can call :class:`ResearchCrewServer` the same way.
"""

from __future__ import annotations

import copy
import threading

from .server import ResearchCrewServer


class AgentMesh:
    """Register A2A servers and route a text message to one by card name."""

    def __init__(self) -> None:
        self._servers: dict[str, ResearchCrewServer] = {}
        self._lock = threading.Lock()

    def register(self, server: ResearchCrewServer) -> None:
        name = server.card["name"]
        with self._lock:
            if name in self._servers:
                raise ValueError(f"agent {name!r} is already on the mesh")
            self._servers[name] = server

    def names(self) -> list[str]:
        with self._lock:
            return list(self._servers)

    def discover(self) -> list[dict]:
        with self._lock:
            return [copy.deepcopy(server.card) for server in self._servers.values()]

    def get(self, name: str) -> ResearchCrewServer:
        with self._lock:
            try:
                return self._servers[name]
            except KeyError:
                raise KeyError(f"no agent named {name!r} on the mesh") from None

    def send(self, name: str, text: str, *, request_id: int | str = 1) -> dict:
        server = self.get(name)
        return server.handle(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "messageId": f"m-{request_id}",
                        "kind": "message",
                        "parts": [{"kind": "text", "text": text}],
                    }
                },
            }
        )


_default: AgentMesh | None = None


def default_mesh() -> AgentMesh:
    """Process-wide mesh. Created on first use."""
    global _default
    if _default is None:
        _default = AgentMesh()
    return _default


def reset_default_mesh() -> AgentMesh:
    """Replace the process-wide mesh. Tests use this to stay isolated."""
    global _default
    _default = AgentMesh()
    return _default


def publish_research_crew(
    mesh: AgentMesh | None = None,
    llm: object | None = None,
    *,
    backend: str = "builtin",
    url: str = "http://127.0.0.1:8089/",
) -> ResearchCrewServer:
    """Build the research crew's A2A server and register it on ``mesh``.

    ``mesh`` defaults to :func:`default_mesh`. This is how the crew joins the
    mesh: by agent card, under the name ``Research Crew``.
    """
    server = ResearchCrewServer(llm=llm, backend=backend, url=url)
    target = mesh if mesh is not None else default_mesh()
    target.register(server)
    return server
