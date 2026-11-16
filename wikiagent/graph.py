"""The agent only sees this interface, so another link graph can replace Wikipedia.

A citation graph fits the same shape: resolve = find paper, links = references,
summary = abstract, backlinks = cited-by. That is the follow-up use case from
the proposal feedback. agent.py, ranking.py and checker.py need no changes.
"""
from dataclasses import dataclass, field
from typing import Protocol


class ToolError(Exception):
    """A read tool failed after its retries."""


class Halt(ToolError):
    """More requests will not help, so the run stops instead of backing up."""


class BudgetExceeded(Halt):
    """The run used up its network budget."""


class RateLimited(Halt):
    """The server asked for a longer wait than we are willing to sleep."""


@dataclass
class Page:
    title: str
    exists: bool = True
    disambiguation: bool = False
    revid: int | None = None
    aliases: list[str] = field(default_factory=list)  # titles that redirect here


class Graph(Protocol):
    def resolve(self, title: str) -> Page: ...
    def links(self, title: str) -> list[str]: ...
    def summary(self, title: str) -> str: ...
    def backlinks(self, title: str) -> list[str]: ...
