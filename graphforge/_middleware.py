"""State middleware — pre/post processing hooks for state transitions.

Provides :class:`StateMiddleware` protocol and :class:`MiddlewarePipeline`
for global interceptors on state updates.

Usage::

    from graphforge._middleware import MiddlewarePipeline, StateMiddleware

    class LoggerMiddleware:
        def pre_update(self, node, state, updates):
            print(f"{node}: updating {list(updates.keys())}")
            return updates

        def post_update(self, node, old_state, new_state):
            print(f"{node}: state updated")

    pipeline = MiddlewarePipeline([LoggerMiddleware()])
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, Sequence

from graphforge._logging import get_logger

logger = get_logger("middleware")


class StateMiddleware(Protocol):
    """Protocol for state middleware hooks.

    Both methods are optional — implement only what you need.
    """

    def pre_update(
        self,
        node: str,
        state: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Called **before** state updates are applied.

        Parameters
        ----------
        node:
            The name of the node that produced the updates.
        state:
            The current state dict (before updates).
        updates:
            The updates dict produced by the node.

        Returns
        -------
        The (possibly modified) updates dict to apply.
        """
        return updates

    def post_update(
        self,
        node: str,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
    ) -> None:
        """Called **after** state updates have been applied.

        Parameters
        ----------
        node:
            The name of the node that produced the updates.
        old_state:
            The state dict before updates.
        new_state:
            The state dict after updates.
        """
        pass


class MiddlewarePipeline:
    """Composes multiple :class:`StateMiddleware` instances into a pipeline.

    Parameters
    ----------
    stages:
        Ordered list of middleware instances.
    """

    def __init__(self, stages: Optional[Sequence[StateMiddleware]] = None) -> None:
        self._stages = list(stages or [])

    def add(self, middleware: StateMiddleware) -> None:
        """Add a middleware stage to the pipeline."""
        self._stages.append(middleware)

    def pre_update(
        self,
        node: str,
        state: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all middleware pre_update hooks in order."""
        current_updates = updates
        for stage in self._stages:
            if hasattr(stage, "pre_update"):
                try:
                    result = stage.pre_update(node, state, current_updates)
                    if result is not None:
                        current_updates = result
                except Exception as e:
                    logger.warning("Middleware pre_update error: %s", e)
        return current_updates

    def post_update(
        self,
        node: str,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
    ) -> None:
        """Run all middleware post_update hooks in order."""
        for stage in self._stages:
            if hasattr(stage, "post_update"):
                try:
                    stage.post_update(node, old_state, new_state)
                except Exception as e:
                    logger.warning("Middleware post_update error: %s", e)


__all__ = [
    "MiddlewarePipeline",
    "StateMiddleware",
]
