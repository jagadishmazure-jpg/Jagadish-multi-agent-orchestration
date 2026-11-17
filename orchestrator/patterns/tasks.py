"""Multi-step tasks shared by ReAct, plan-and-execute, and ReWOO.

Each task needs more than one tool call. The plan-and-execute scripts start
with an incomplete plan so the replanner has to add steps. ReWOO states every
dependency up front with ``#E`` references. ReAct discovers the same tools one
call at a time. Answers are exact strings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlTask:
    """One comparison task and the scripted model behaviour for each pattern."""

    id: str
    question: str
    answer: str
    initial_plan: tuple[str, ...]
    replans: dict[int, tuple[str, ...] | str]
    rewoo_lines: tuple[str, ...]
    react_steps: tuple[tuple[str, dict] | None, ...]


FACTS: dict[str, str] = {
    "capital of france": "Paris",
    "capital of japan": "Tokyo",
    "capital of australia": "Canberra",
    "population of paris": "2100000",
    "population of tokyo": "14000000",
    "population of canberra": "470000",
    "creator of python": "Guido van Rossum",
    "release year of python": "1991",
}


TASKS: tuple[ControlTask, ...] = (
    ControlTask(
        id="capitals_population_sum",
        question="What is the sum of the populations of the capitals of France and Japan?",
        answer="16100000",
        initial_plan=(
            "search: capital of France",
            "search: capital of Japan",
        ),
        replans={
            1: ("search: capital of Japan",),
            2: (
                "search: population of Paris",
                "search: population of Tokyo",
            ),
            3: ("search: population of Tokyo",),
            4: ("calculator: 2100000+14000000",),
            5: "FINAL: 16100000",
        },
        rewoo_lines=(
            "#E1 = search[capital of France]",
            "#E2 = search[capital of Japan]",
            "#E3 = search[population of #E1]",
            "#E4 = search[population of #E2]",
            "#E5 = calculator[#E3+#E4]",
        ),
        react_steps=(
            ("search", {"query": "capital of France"}),
            ("search", {"query": "capital of Japan"}),
            ("search", {"query": "population of Paris"}),
            ("search", {"query": "population of Tokyo"}),
            ("calculator", {"expression": "2100000+14000000"}),
            None,
        ),
    ),
    ControlTask(
        id="canberra_twice",
        question="What is twice the population of the capital of Australia?",
        answer="940000",
        initial_plan=("search: capital of Australia",),
        replans={
            1: ("search: population of Canberra",),
            2: ("calculator: 470000*2",),
            3: "FINAL: 940000",
        },
        rewoo_lines=(
            "#E1 = search[capital of Australia]",
            "#E2 = search[population of #E1]",
            "#E3 = calculator[#E2*2]",
        ),
        react_steps=(
            ("search", {"query": "capital of Australia"}),
            ("search", {"query": "population of Canberra"}),
            ("calculator", {"expression": "470000*2"}),
            None,
        ),
    ),
    ControlTask(
        id="python_identity",
        question="Who created Python and in what year was it first released? Reply as 'Name, year'.",
        answer="Guido van Rossum, 1991",
        initial_plan=(
            "search: creator of Python",
            "search: release year of Python",
        ),
        replans={
            1: ("search: release year of Python",),
            2: "FINAL: Guido van Rossum, 1991",
        },
        rewoo_lines=(
            "#E1 = search[creator of Python]",
            "#E2 = search[release year of Python]",
        ),
        react_steps=(
            ("search", {"query": "creator of Python"}),
            ("search", {"query": "release year of Python"}),
            None,
        ),
    ),
)


def task_by_id(task_id: str) -> ControlTask:
    for task in TASKS:
        if task.id == task_id:
            return task
    raise KeyError(task_id)


def lookup_fact(query: str) -> str:
    """Longest corpus key that is contained in the query, else a miss."""
    lowered = query.lower()
    found = [key for key in FACTS if key in lowered]
    if not found:
        return f"No results for {query!r}"
    return FACTS[max(found, key=len)]
