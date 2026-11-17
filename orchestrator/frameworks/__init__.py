"""Same research-and-report task on CrewAI, LangGraph, and Agent Framework."""

from .compare import FrameworkScore, compare_frameworks, count_code_lines
from .crewai_research import run_crewai_research
from .golden import load_cases
from .langgraph_research import run_langgraph_research
from .maf_research import run_maf_research
from .mock import ResearchMockLLM

__all__ = [
    "FrameworkScore",
    "ResearchMockLLM",
    "compare_frameworks",
    "count_code_lines",
    "load_cases",
    "run_crewai_research",
    "run_langgraph_research",
    "run_maf_research",
]
