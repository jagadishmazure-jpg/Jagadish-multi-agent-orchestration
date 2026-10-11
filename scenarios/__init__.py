"""End-to-end offline scenarios composing agents, patterns, and tools."""

from .research_team import build_research_team, run_research_scenario
from .support_triage import build_triage_system, triage_ticket

__all__ = [
    "build_research_team",
    "build_triage_system",
    "run_research_scenario",
    "triage_ticket",
]
