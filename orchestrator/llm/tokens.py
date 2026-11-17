"""Deterministic token estimates shared by the mock meter and the Azure client.

The estimate is the usual one-token-per-four-characters rounding. It is not a
model tokenizer. It is stable, so comparison tables do not move between runs.
"""

from __future__ import annotations


def count_text(text: str) -> int:
    """Return the estimated token count for ``text`` (0 for an empty string)."""
    if not text:
        return 0
    return (len(text) + 3) // 4
