"""Golden research-and-report cases loaded from ``evals/golden_research.json``."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

_REQUIRED = ("id", "topic", "match", "keywords", "plan", "findings", "draft", "review")


def golden_path() -> Path:
    """Find the golden set in the repo checkout or next to this module."""
    candidates = (
        Path(__file__).resolve().parents[2] / "evals" / "golden_research.json",
        Path(__file__).resolve().parent / "golden_research.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("golden research set not found at " + ", ".join(str(path) for path in candidates))


@cache
def load_cases() -> tuple[dict, ...]:
    data = json.loads(golden_path().read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) < 3:
        raise ValueError("golden research set needs at least 3 cases")
    checked: list[dict] = []
    for case in cases:
        missing = [key for key in _REQUIRED if key not in case]
        if missing:
            raise ValueError(f"golden case {case.get('id', '?')!r} missing {missing}")
        checked.append(case)
    return tuple(checked)


def match_case(text: str) -> dict | None:
    """Pick the case whose match token appears in ``text``. Longest token wins."""
    lowered = text.lower()
    found = [
        case
        for case in load_cases()
        if any(token.lower() in lowered for token in case["match"])
    ]
    if not found:
        return None
    return max(found, key=lambda case: max(len(token) for token in case["match"]))


def case_by_id(case_id: str) -> dict | None:
    for case in load_cases():
        if case["id"] == case_id:
            return case
    return None


def quality_score(final: str, case: dict | None) -> float:
    """Fraction of required keywords present. Missing APPROVED scores zero."""
    if not case:
        return 0.0
    if "APPROVED" not in final:
        return 0.0
    keywords = list(case["keywords"])
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in final)
    return hits / len(keywords)
