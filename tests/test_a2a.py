"""A2A card, JSON-RPC, and the in-process mesh the research crew joins."""

import json
import urllib.request

import pytest

from orchestrator.a2a import (
    AGENT_CARD_PATHS,
    AgentMesh,
    publish_research_crew,
    research_crew_card,
    reset_default_mesh,
    serve,
)
from orchestrator.a2a.server import ResearchCrewServer
from orchestrator.frameworks.mock import ResearchMockLLM


def _send(text: str, method: str = "message/send", request_id: int = 1) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            "message": {
                "role": "user",
                "messageId": "m1",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }


def test_agent_card_advertises_the_research_skill():
    card = research_crew_card()
    assert card["name"] == "Research Crew"
    assert card["skills"][0]["id"] == "research-and-report"
    assert "/.well-known/agent-card.json" in AGENT_CARD_PATHS


def test_message_send_and_get_return_the_report():
    server = ResearchCrewServer(llm=ResearchMockLLM(), backend="builtin")
    response = server.handle(_send("Research: The history of Python"))
    task = response["result"]
    assert response["id"] == 1
    assert task["status"]["state"] == "completed"
    report = task["artifacts"][0]["parts"][0]["text"]
    assert "Guido van Rossum" in report
    assert "APPROVED" in report
    fetched = server.handle(
        {"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": task["id"]}}
    )
    assert fetched["result"]["artifacts"] == task["artifacts"]
    alias = server.handle(_send("Research: The origins of HTTP", method="SendMessage", request_id=3))
    assert "Tim Berners-Lee" in alias["result"]["artifacts"][0]["parts"][0]["text"]


def test_jsonrpc_errors_and_cancel_rules():
    server = ResearchCrewServer(llm=ResearchMockLLM())
    assert server.handle([])["error"]["code"] == -32600
    assert server.handle({"jsonrpc": "2.0", "id": 1, "method": "nope"})["error"]["code"] == -32601
    assert server.handle(_send(""))["error"]["code"] == -32602
    missing = server.handle(
        {"jsonrpc": "2.0", "id": 4, "method": "tasks/cancel", "params": {"id": "task-missing"}}
    )
    assert missing["error"]["code"] == -32602
    created = server.handle(_send("Research: The relational database model"))
    cancelled = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tasks/cancel",
            "params": {"id": created["result"]["id"]},
        }
    )
    assert "cannot be cancelled" in cancelled["error"]["message"]
    server.tasks["task-9"] = {"id": "task-9", "status": {"state": "working"}}
    stopped = server.handle(
        {"jsonrpc": "2.0", "id": 6, "method": "CancelTask", "params": {"taskId": "task-9"}}
    )
    assert stopped["result"]["status"]["state"] == "canceled"


def test_crew_joins_the_mesh_and_duplicate_registration_fails():
    mesh = AgentMesh()
    publish_research_crew(mesh, llm=ResearchMockLLM(), backend="builtin")
    assert mesh.names() == ["Research Crew"]
    assert mesh.discover()[0]["skills"][0]["id"] == "research-and-report"
    response = mesh.send("Research Crew", "Research: The history of Python")
    assert response["result"]["status"]["state"] == "completed"
    with pytest.raises(ValueError, match="already on the mesh"):
        publish_research_crew(mesh, llm=ResearchMockLLM())


def test_default_mesh_can_be_reset_and_joined():
    first = reset_default_mesh()
    publish_research_crew(llm=ResearchMockLLM())
    assert first.names() == ["Research Crew"]
    second = reset_default_mesh()
    assert second.names() == []
    assert first.names() == ["Research Crew"]


def test_http_serves_the_card_and_a_message():
    server = ResearchCrewServer(llm=ResearchMockLLM(), backend="builtin")
    httpd = serve(server, port=0)
    try:
        port = httpd.server_address[1]
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(f"{base}/.well-known/agent-card.json", timeout=5) as response:
            card = json.loads(response.read().decode())
        assert card["name"] == "Research Crew"
        assert card["url"] == f"{base}/"
        body = json.dumps(_send("Research: The history of Python")).encode()
        request = urllib.request.Request(
            f"{base}/rpc",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode())
        assert "1991" in payload["result"]["artifacts"][0]["parts"][0]["text"]
    finally:
        httpd.shutdown()
        httpd.server_close()
