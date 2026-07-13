"""Linear pipeline for sequential processing.

A :class:`Pipeline` is a lightweight, ordered sequence of steps that
process state in order. It is **not** a graph — there is no branching,
no routing, and no cycles. Each step feeds its output directly into the
next.

Pipelines are useful for simple LLM call chains, data preprocessing, or
any workflow that can be expressed as a linear sequence. They also serve
as nodes inside graphs via :meth:`Graph.add_node`.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Union,
)

from graphforge._logging import get_logger
from graphforge._types import (
    AsyncNodeFunc,
    NodeFunc,
    StateT,
    StateUpdate,
)

logger = get_logger("pipeline")


class Pipeline(Generic[StateT]):
    """An ordered sequence of processing steps.

    Parameters
    ----------
    steps:
        Sequence of callables ``(state, **kwargs) -> dict``. Each step
        receives the *accumulated* state.
    name:
        Optional human-readable name.
    """

    __slots__ = ("_steps", "_name", "_has_async")

    def __init__(
        self,
        steps: Sequence[Union[NodeFunc, AsyncNodeFunc]],
        name: Optional[str] = None,
    ) -> None:
        if not steps:
            raise ValueError("A Pipeline must have at least one step.")
        self._steps = list(steps)
        self._name = name or "unnamed_pipeline"
        self._has_async = self._detect_async()
        logger.debug("Pipeline %r created with %d steps", self._name, len(self._steps))

    @property
    def name(self) -> str:
        return self._name

    @property
    def steps(self) -> List[Union[NodeFunc, AsyncNodeFunc]]:
        return list(self._steps)

    def is_async(self) -> bool:
        return self._has_async

    # -- execution --------------------------------------------------------

    def run(
        self,
        initial_state: StateT,
        **kwargs: Any,
    ) -> StateUpdate:
        """Execute the pipeline synchronously."""
        if self._has_async:
            raise TypeError(
                f"Pipeline {self._name!r} contains async steps; "
                f"use ``arun()`` instead."
            )

        accumulated: Dict[str, Any] = _dump(initial_state)
        logger.info("Pipeline %r: running %d steps", self._name, len(self._steps))

        for i, step in enumerate(self._steps):
            state_obj = _reify(accumulated, initial_state)
            logger.debug("Pipeline %r step %d/%d", self._name, i + 1, len(self._steps))
            updates = step(state_obj, **kwargs)
            logger.debug("Step %d produced: %s", i + 1, list(updates.keys()))
            accumulated.update(updates)

        logger.info("Pipeline %r finished", self._name)
        return accumulated

    async def arun(
        self,
        initial_state: StateT,
        **kwargs: Any,
    ) -> StateUpdate:
        """Execute the pipeline asynchronously."""
        accumulated: Dict[str, Any] = _dump(initial_state)
        logger.info("Pipeline %r: async running %d steps", self._name, len(self._steps))

        for i, step in enumerate(self._steps):
            state_obj = _reify(accumulated, initial_state)
            logger.debug("Pipeline %r async step %d/%d", self._name, i + 1, len(self._steps))
            if _is_async(step):
                updates = await step(state_obj, **kwargs)
            else:
                updates = step(state_obj, **kwargs)
            logger.debug("Async step %d produced: %s", i + 1, list(updates.keys()))
            accumulated.update(updates)

        logger.info("Pipeline %r finished", self._name)
        return accumulated

    def _detect_async(self) -> bool:
        return any(_is_async(step) for step in self._steps)

    def __repr__(self) -> str:
        return (
            f"Pipeline(name={self._name!r}, "
            f"steps={len(self._steps)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from graphforge._executor import _dump  # noqa: E402


def _reify(
    accumulated: Dict[str, Any],
    prototype: Any,
) -> Any:
    if hasattr(prototype, "model_validate"):
        return type(prototype).model_validate(accumulated)
    return accumulated


def _is_async(fn: Any) -> bool:
    import inspect
    return inspect.iscoroutinefunction(fn)


__all__ = [
    "Pipeline",
]
