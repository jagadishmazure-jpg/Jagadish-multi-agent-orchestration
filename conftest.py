"""Shared pytest fixtures for the offline test suite."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from orchestrator.core import Agent, TaskState  # noqa: E402
from orchestrator.llm import FakeLLM  # noqa: E402


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(default="ok")


@pytest.fixture
def task_state() -> TaskState:
    return TaskState(task="test task")


@pytest.fixture
def simple_agent(fake_llm: FakeLLM) -> Agent:
    return Agent(name="worker", role="test worker", llm=fake_llm)
