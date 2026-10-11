"""Support-triage scenario: classify a ticket, then route to a specialist.

Specialists use the offline tool loop: billing computes refunds with the
calculator, technical looks up known issues with the search stub, and account
answers directly. Fully offline and deterministic.
"""

from __future__ import annotations

from orchestrator.core import Agent
from orchestrator.eval import TrajectoryRecorder
from orchestrator.llm import FakeLLM
from orchestrator.patterns import Router
from orchestrator.tools import (
    ToolCallingLoop,
    ToolRegistry,
    make_calculator_tool,
    make_search_stub_tool,
)


def _classifier() -> Agent:
    return Agent(
        name="triage_classifier",
        role="ticket classifier",
        llm=FakeLLM(
            rules=[
                ("bill", "CATEGORY: billing"),
                ("invoice", "CATEGORY: billing"),
                ("charge", "CATEGORY: billing"),
                ("refund", "CATEGORY: billing"),
                ("error", "CATEGORY: technical"),
                ("crash", "CATEGORY: technical"),
                ("login", "CATEGORY: technical"),
                ("bug", "CATEGORY: technical"),
                ("password", "CATEGORY: account"),
                ("account", "CATEGORY: account"),
                ("profile", "CATEGORY: account"),
            ],
            default="CATEGORY: general",
        ),
    )


def build_triage_system(
    recorder: TrajectoryRecorder | None = None,
) -> tuple[Router, dict[str, ToolRegistry]]:
    """Build the router plus per-specialist tool registries."""
    recorder = recorder if recorder is not None else TrajectoryRecorder()

    billing_registry = ToolRegistry()
    billing_registry.register(make_calculator_tool())
    billing = Agent(
        name="billing_specialist",
        role="billing support specialist",
        llm=FakeLLM(
            sequence=[
                {
                    "content": "I'll compute the refund for 3 months.",
                    "tool_calls": [
                        {
                            "name": "calculator",
                            "arguments": {"expression": "29.99*3"},
                            "call_id": "call_1",
                        }
                    ],
                },
                "RESOLUTION: a refund of $89.97 will be credited to the "
                "original payment method within 5-7 days.",
            ]
        ),
        tools=[make_calculator_tool()],
    )

    technical_registry = ToolRegistry()
    technical_registry.register(
        make_search_stub_tool(
            {
                "login error": (
                    "Known issue KB-1042: login errors after password reset; "
                    "clear cookies and retry."
                )
            }
        )
    )
    technical = Agent(
        name="technical_specialist",
        role="technical support specialist",
        llm=FakeLLM(
            sequence=[
                {
                    "content": "Looking up the reported error.",
                    "tool_calls": [
                        {
                            "name": "search",
                            "arguments": {"query": "login error"},
                            "call_id": "call_1",
                        }
                    ],
                },
                "RESOLUTION: this matches known issue KB-1042. Please clear "
                "cookies and retry logging in.",
            ]
        ),
        tools=[make_search_stub_tool()],
    )

    account = Agent(
        name="account_specialist",
        role="account support specialist",
        llm=FakeLLM(
            default=(
                "RESOLUTION: your account email has been updated; a "
                "confirmation link is on its way."
            )
        ),
    )

    router = Router(
        classifier=_classifier(),
        specialists={
            "billing": billing,
            "technical": technical,
            "account": account,
        },
        recorder=recorder,
    )
    registries = {"billing": billing_registry, "technical": technical_registry}
    return router, registries


def triage_ticket(ticket: str) -> dict:
    """Classify a ticket and resolve it with the matching specialist."""
    recorder = TrajectoryRecorder()
    router, registries = build_triage_system(recorder=recorder)
    routed = router.run(ticket)
    specialist = router.specialists[routed.category]
    if specialist.name in ("billing_specialist", "technical_specialist"):
        loop = ToolCallingLoop(
            specialist,
            registries[routed.category],
            max_iterations=4,
            recorder=recorder,
        )
        final = loop.run(ticket)
        output = final.content
    else:
        output = routed.output
    return {
        "ticket": ticket,
        "category": routed.category,
        "agent": routed.agent_name,
        "output": output,
        "trajectory": recorder,
    }
