"""A2A JSON-RPC surface for the CrewAI research crew.

The repo did not already contain an agent mesh. This server publishes an
agent card and answers ``message/send``, ``tasks/get``, and ``tasks/cancel``
(and the PascalCase aliases from the later spec) so a mesh that speaks A2A
can call the crew. :class:`orchestrator.a2a.mesh.AgentMesh` is the in-process
mesh the crew joins.
"""

from __future__ import annotations

import copy
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..frameworks.crewai_research import run_crewai_research

AGENT_CARD_PATHS = (
    "/.well-known/agent-card.json",
    "/.well-known/agent.json",
)
RPC_PATHS = ("/", "/rpc", "/a2a")

_SEND = {"message/send", "SendMessage"}
_GET = {"tasks/get", "GetTask"}
_CANCEL = {"tasks/cancel", "CancelTask"}
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_MAX_BODY = 64 * 1024
_READ_TIMEOUT_S = 5
_DEFAULT_TASKS = 32


def research_crew_card(url: str = "http://127.0.0.1:8089/") -> dict[str, Any]:
    """Agent card for the research-and-report crew."""
    return {
        "name": "Research Crew",
        "description": (
            "CrewAI crew that plans research, gathers findings, drafts a report, and reviews it."
        ),
        "protocolVersion": "0.3.0",
        "version": "0.1.0",
        "url": url,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "research-and-report",
                "name": "Research and report",
                "description": "Research a topic and return a critic-reviewed article.",
                "tags": ["research", "crewai", "report"],
                "examples": ["Research: The history of Python"],
            }
        ],
    }


class ResearchCrewServer:
    """In-process A2A server. ``handle`` is the JSON-RPC entry point."""

    def __init__(
        self,
        llm: Any = None,
        *,
        backend: str = "builtin",
        url: str = "http://127.0.0.1:8089/",
        max_tasks: int = _DEFAULT_TASKS,
    ) -> None:
        # ``llm is None`` lets the crew runner pick Azure OpenAI when that
        # environment is fully set, and the scripted mock otherwise.
        if max_tasks < 1:
            raise ValueError("max_tasks must be >= 1")
        self.llm = llm
        self.backend = backend
        self.card = research_crew_card(url)
        self.max_tasks = max_tasks
        self.tasks: dict[str, dict] = {}
        self._order: list[str] = []
        self._by_request: dict[Any, str] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def handle(self, request: Any) -> dict:
        if not isinstance(request, dict):
            return _error(None, -32600, "invalid request")
        request_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str) or not method:
            return _error(request_id, -32600, "invalid request")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            return _error(request_id, -32602, "invalid params")
        if method in _SEND:
            return self._send(request_id, params)
        if method in _GET:
            return self._get(request_id, params)
        if method in _CANCEL:
            return self._cancel(request_id, params)
        return _error(request_id, -32601, f"method not found: {method}")

    def _send(self, request_id: Any, params: dict) -> dict:
        with self._lock:
            cached = self._cached(request_id)
        if cached is not None:
            return cached
        text = message_text(params)
        if not text:
            return _error(request_id, -32602, "message has no text part")
        try:
            run = run_crewai_research(text, llm=self.llm, backend=self.backend)
        except ValueError as exc:
            return _error(request_id, -32602, str(exc))
        except Exception:
            return _error(request_id, -32000, "research crew failed")
        # message/send finishes before it returns, so the caller receives a
        # completed task. Cancel applies to a task still marked working.
        with self._lock:
            cached = self._cached(request_id)
            if cached is not None:
                return cached
            self._seq += 1
            number = self._seq
            report = run.stages.get("draft") or run.final
            review = run.stages.get("review") or run.final
            task = {
                "id": f"task-{number}",
                "contextId": f"ctx-{number}",
                "status": {"state": "completed"},
                "history": [
                    {
                        "role": "user",
                        "messageId": f"user-{number}",
                        "parts": [{"kind": "text", "text": text}],
                    },
                    {
                        "role": "agent",
                        "messageId": f"agent-{number}",
                        "parts": [{"kind": "text", "text": report}],
                    },
                ],
                "artifacts": [
                    {
                        "artifactId": f"report-{number}",
                        "name": "research-report",
                        "parts": [{"kind": "text", "text": report}],
                    },
                    {
                        "artifactId": f"review-{number}",
                        "name": "critic-review",
                        "parts": [{"kind": "text", "text": review}],
                    },
                ],
            }
            self._remember(task, request_id)
            return {"jsonrpc": "2.0", "id": request_id, "result": copy.deepcopy(task)}

    def _cached(self, request_id: Any) -> dict | None:
        if request_id is None or request_id not in self._by_request:
            return None
        task = self.tasks.get(self._by_request[request_id])
        if task is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": copy.deepcopy(task)}

    def _remember(self, task: dict, request_id: Any) -> None:
        self.tasks[task["id"]] = task
        self._order.append(task["id"])
        if request_id is not None:
            self._by_request[request_id] = task["id"]
        while len(self._order) > self.max_tasks:
            dropped = self._order.pop(0)
            self.tasks.pop(dropped, None)
            stale = [key for key, value in self._by_request.items() if value == dropped]
            for key in stale:
                del self._by_request[key]

    def _get(self, request_id: Any, params: dict) -> dict:
        with self._lock:
            task = self.tasks.get(_task_id(params))
            if task is None:
                return _error(request_id, -32602, "task not found")
            return {"jsonrpc": "2.0", "id": request_id, "result": copy.deepcopy(task)}

    def _cancel(self, request_id: Any, params: dict) -> dict:
        with self._lock:
            task_id = _task_id(params)
            task = self.tasks.get(task_id)
            if task is None:
                return _error(request_id, -32602, "task not found")
            state = (task.get("status") or {}).get("state")
            if state in {"completed", "canceled", "failed"}:
                return _error(request_id, -32602, f"task {task_id} is {state} and cannot be cancelled")
            task["status"] = {"state": "canceled"}
            return {"jsonrpc": "2.0", "id": request_id, "result": copy.deepcopy(task)}


def message_text(params: dict) -> str:
    """Pull text parts out of a ``message/send`` params object."""
    message = params.get("message") or {}
    if isinstance(message, dict) and isinstance(message.get("message"), dict):
        message = message["message"]
    if not isinstance(message, dict):
        return ""
    texts: list[str] = []
    for part in message.get("parts") or []:
        if isinstance(part, dict) and "text" in part:
            kind = part.get("kind", part.get("type", "text"))
            if kind in (None, "text"):
                texts.append(str(part["text"]))
    if not texts and isinstance(message.get("text"), str):
        texts.append(message["text"])
    return "\n".join(texts).strip()


def _task_id(params: dict) -> str:
    for key in ("id", "taskId", "task_id"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class _Handler(BaseHTTPRequestHandler):
    """HTTP binding: GET the agent card, POST JSON-RPC."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        path = self.path.split("?", 1)[0]
        if path in AGENT_CARD_PATHS:
            card = getattr(self.server, "card", None)
            if card is None:
                crew: ResearchCrewServer = self.server.a2a  # type: ignore[attr-defined]
                card = crew.card
            self._send(200, card)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        path = self.path.split("?", 1)[0]
        if path not in RPC_PATHS:
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            self._send(400, _error(None, -32700, "invalid content length"))
            return
        if length < 0 or length > _MAX_BODY:
            self._send(413, _error(None, -32600, "request body too large"))
            return
        self.connection.settimeout(_READ_TIMEOUT_S)
        try:
            raw = self.rfile.read(length) if length else b""
        except (TimeoutError, socket.timeout):
            self._send(408, _error(None, -32603, "request timed out"))
            return
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except json.JSONDecodeError:
            self._send(400, _error(None, -32700, "parse error"))
            return
        server: ResearchCrewServer = self.server.a2a  # type: ignore[attr-defined]
        try:
            result = server.handle(payload)
        except Exception:
            request_id = payload.get("id") if isinstance(payload, dict) else None
            result = _error(request_id, -32000, "internal error")
        self._send(200, result)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(
    server: ResearchCrewServer | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    advertise_host: str | None = None,
) -> ThreadingHTTPServer:
    """Start the crew's A2A HTTP server on ``port`` (0 picks a free port).

    The HTTP binding is loopback only. It is a local demo, not a networked
    service, so it does not accept a public bind address. Each listener keeps
    its own agent card, so starting a second listener does not change the URL
    the first one advertises. ``llm`` defaults to ``None`` on a new server,
    which selects Azure OpenAI when that environment is fully set.

    The returned server is already running on a daemon thread. Call
    ``shutdown`` and ``server_close`` when finished.
    """
    if host not in _LOOPBACK:
        raise ValueError(
            "the A2A demo server only binds to loopback (127.0.0.1, localhost, or ::1)"
        )
    shown = advertise_host or ("127.0.0.1" if host == "localhost" else host)
    if shown in {"0.0.0.0", "::", ""}:
        raise ValueError("advertise_host must be a client-reachable address")
    crew = server if server is not None else ResearchCrewServer()
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.a2a = crew  # type: ignore[attr-defined]
    bound = httpd.server_address[1]
    httpd.card = research_crew_card(f"http://{shown}:{bound}/")  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd
