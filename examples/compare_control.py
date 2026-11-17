"""Print ReAct vs plan-and-execute vs ReWOO on the shared multi-step tasks.

Mock numbers are deterministic. ``--live`` records Azure OpenAI when configured.
"""

from __future__ import annotations

import sys

from orchestrator.llm import MOCK_CALL_MS, AzureOpenAIClient, azure_configured
from orchestrator.patterns import compare_control
from orchestrator.patterns.control_llm import ControlMockLLM

HEADERS = (
    "pattern",
    "success_rate",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_ms",
    "llm_calls",
    "tasks",
    "runtime",
)
LIVE_HEADERS = ("pattern", "success_rate", "total_tokens", "latency_ms", "tasks")


def main() -> None:
    scores = compare_control(llm=ControlMockLLM(), backend="builtin")
    print(f"Control-pattern comparison (mock LLM, deterministic latency = {MOCK_CALL_MS} ms/call)")
    print("patterns: react (existing tool loop), plan_execute (planner/executor/replanner), rewoo")
    rows = [
        [
            score.pattern,
            f"{score.success_rate:.2f}",
            str(score.prompt_tokens),
            str(score.completion_tokens),
            str(score.total_tokens),
            str(score.latency_ms),
            str(score.llm_calls),
            f"{score.passed}/{score.tasks}",
            score.runtime,
        ]
        for score in scores
    ]
    print(format_table(HEADERS, rows))
    print()
    if "--live" in sys.argv and azure_configured():
        live = compare_control(llm=AzureOpenAIClient.from_env(), backend="builtin")
        print("Live results (Azure OpenAI, wall-clock latency):")
        live_rows = [
            [
                score.pattern,
                f"{score.success_rate:.2f}",
                str(score.total_tokens),
                str(score.latency_ms),
                f"{score.passed}/{score.tasks}",
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
    print("  python examples/compare_control.py --live")
    blank = [
        ["react", "-", "-", "-", "-"],
        ["plan_execute", "-", "-", "-", "-"],
        ["rewoo", "-", "-", "-", "-"],
    ]
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
