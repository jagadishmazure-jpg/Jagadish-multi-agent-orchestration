"""Shared pytest fixtures for the offline test suite."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from orchestrator.core import Agent, TaskState  # noqa: E402
from orchestrator.llm import FakeLLM  # noqa: E402

_AZURE_ENV = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
)


@pytest.fixture(autouse=True)
def _offline_azure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite on the mock even if the shell has Azure credentials."""
    for key in _AZURE_ENV:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(default="ok")


@pytest.fixture
def task_state() -> TaskState:
    return TaskState(task="test task")


@pytest.fixture
def simple_agent(fake_llm: FakeLLM) -> Agent:
    return Agent(name="worker", role="test worker", llm=fake_llm)
