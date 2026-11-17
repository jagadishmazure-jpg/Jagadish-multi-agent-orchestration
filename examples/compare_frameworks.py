"""Print the CrewAI / LangGraph / Agent Framework comparison.

The checked table always uses the deterministic mock. ``--live`` fills the
Azure table when AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and
AZURE_OPENAI_DEPLOYMENT are set, and leaves the placeholder otherwise.
"""

from __future__ import annotations

import sys

from orchestrator.frameworks import compare_frameworks
from orchestrator.frameworks.mock import ResearchMockLLM
from orchestrator.llm import MOCK_CALL_MS, AzureOpenAIClient, azure_configured

HEADERS = (
    "framework",
    "quality",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_ms",
    "llm_calls",
    "code_lines",
    "cases",
    "runtime",
)
LIVE_HEADERS = ("framework", "quality", "total_tokens", "latency_ms", "cases")


def main() -> None:
    scores = compare_frameworks(llm=ResearchMockLLM(), backend="builtin")
    print(f"Framework comparison (mock LLM, deterministic latency = {MOCK_CALL_MS} ms/call)")
    print(
        "backend: builtin. The same specs run on crewai, langgraph, and "
        "agent-framework when those packages are installed."
    )
    rows = [
        [
            score.framework,
            f"{score.quality:.2f}",
            str(score.prompt_tokens),
            str(score.completion_tokens),
            str(score.total_tokens),
            str(score.latency_ms),
            str(score.llm_calls),
            str(score.code_lines),
            f"{score.passed}/{score.cases}",
            score.runtime,
        ]
        for score in scores
    ]
    print(format_table(HEADERS, rows))
    print()
    if "--live" in sys.argv and azure_configured():
        live = compare_frameworks(llm=AzureOpenAIClient.from_env(), backend="builtin")
        print("Live results (Azure OpenAI, wall-clock latency):")
        live_rows = [
            [
                score.framework,
                f"{score.quality:.2f}",
                str(score.total_tokens),
                str(score.latency_ms),
                f"{score.passed}/{score.cases}",
            ]
            for score in live
        ]
        print(format_table(LIVE_HEADERS, live_rows))
        return
    reason = (
        "skipped. Azure OpenAI env vars are not set."
        if "--live" in sys.argv
        else "not recorded in this run."
    )
    print(f"Live results (Azure OpenAI): {reason}")
    print(
        "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and "
        "AZURE_OPENAI_DEPLOYMENT, then run:"
    )
    print("  python examples/compare_frameworks.py --live")
    blank = [["crewai", "-", "-", "-", "-"], ["langgraph", "-", "-", "-", "-"], ["agent_framework", "-", "-", "-", "-"]]
    print(format_table(LIVE_HEADERS, blank))


def format_table(headers: tuple[str, ...], rows: list[list[str]]) -> str:
    columns = list(zip(*([list(headers)] + rows), strict=True))
    widths = [max(len(cell) for cell in column) for column in columns]

    def line(items: list[str]) -> str:
        return "  ".join(item.ljust(widths[index]) for index, item in enumerate(items)).rstrip()

    rendered = [line(list(headers)), line(["-" * width for width in widths])]
    rendered.extend(line(row) for row in rows)
    return "\n".join(rendered)


if __name__ == "__main__":
    main()
