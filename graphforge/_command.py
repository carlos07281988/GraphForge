"""Command — dynamic node-level routing directive.

A :class:`Command` allows a node to dynamically control the execution flow
by specifying both state updates and the next node to execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Command:
    """A directive returned by a node to control the next execution step.

    When a node returns a :class:`Command` instead of a plain ``dict``, the
    executor applies the state *update* and routes to *goto* directly,
    bypassing normal edge resolution.

    Args:
        goto:
            The name of the next node to execute.
        update:
            Optional state updates to apply before the next node runs.
    """

    goto: str
    update: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.update is None:
            self.update = {}

    def __repr__(self) -> str:
        return (
            f"Command(goto={self.goto!r}"
            f"{', update=' + str(self.update) if self.update else ''})"
        )


__all__ = ["Command"]
