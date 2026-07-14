"""Interrupt — pause graph execution for human input.

Provides :func:`interrupt` which nodes can call to pause execution and
wait for human input via :meth:`~graphforge._graph.CompiledGraph.resume`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from graphforge._executor import GraphExecutionPaused

_logger = logging.getLogger(__name__)


def interrupt(
    *,
    message: str = "Execution interrupted",
    value: Any = None,
) -> None:
    """Pause graph execution and wait for human input.

    When a node calls this function, the executor saves a checkpoint and
    returns control to the caller.  Use :meth:`~graphforge._graph.CompiledGraph.resume`
    with the same ``thread_id`` to continue execution.

    Args:
        message: Human-readable description of the interruption.
        value: Optional data to pass to the caller (e.g. a question or
            context that helps the human provide input).
    """
    raise GraphExecutionPaused(message)


__all__ = ["interrupt"]
