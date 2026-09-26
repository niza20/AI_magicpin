"""Agent base: a named unit with one responsibility, structured I/O, and a trace entry per run."""
from __future__ import annotations

from typing import Any

from ..types import TraceStep


class Agent:
    name = "agent"
    uses_llm = False

    def __init__(self, trace: list[TraceStep] | None = None) -> None:
        self.trace = trace if trace is not None else []

    def log(self, summary: str, data: Any = None) -> None:
        self.trace.append(TraceStep(agent=self.name, summary=summary, data=data))
