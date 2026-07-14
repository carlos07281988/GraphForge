"""Node definitions for graph construction.

A **node** is the atomic unit of computation in a GraphForge graph. It
receives the current :class:`~graphforge.state.GraphState` and returns a
partial :data:`~graphforge._types.StateUpdate` dict that the framework
folds into the state.
"""

from __future__ import annotations

import inspect
import logging
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Optional,
    Union,
)

from graphforge._logging import get_logger
from graphforge._types import (
    AsyncNodeFunc,
    AsyncStreamingNodeFunc,
    NodeFunc,
    NodeName,
    StateT,
    StateUpdate,
    StreamingNodeFunc,
)

if TYPE_CHECKING:
    from graphforge._graph import CompiledGraph
    from graphforge.pipeline import Pipeline

logger = get_logger("node")


# ---------------------------------------------------------------------------
# NodeKind — enumerates the internal execution paths
# ---------------------------------------------------------------------------


class NodeKind(str, Enum):
    """Runtime classification of a node's invocation path."""

    FUNCTION = "function"
    ASYNC = "async"
    STREAM = "stream"
    ASYNC_STREAM = "async_stream"
    SUBGRAPH = "subgraph"
    PIPELINE = "pipeline"


# ---------------------------------------------------------------------------
# Node — wraps any callable with metadata
# ---------------------------------------------------------------------------


class Node(Generic[StateT]):
    """A named node in a graph.

    Parameters
    ----------
    name:
        Unique node identifier within the graph.
    fn:
        The callable that implements the node's logic.
    retry:
        Number of retry attempts on failure (default: 0 = no retry).
    timeout:
        Maximum execution time in seconds (default: None = no timeout).
    metadata:
        Optional arbitrary metadata (tags, version, etc.).
    """

    __slots__ = ("_name", "_fn", "_kind", "_retry", "_timeout", "_metadata")

    def __init__(
        self,
        name: NodeName,
        fn: Union[
            NodeFunc,
            AsyncNodeFunc,
            StreamingNodeFunc,
            AsyncStreamingNodeFunc,
            "CompiledGraph[StateT]",
            "Pipeline[StateT]",
        ],
        *,
        retry: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name = name
        self._fn = fn
        self._kind = _classify(fn)
        self._retry = retry
        self._timeout = timeout
        self._metadata = metadata or {}
        logger.debug("Node %r: kind=%s", name, self._kind.value)

    # -- read-only properties -----------------------------------------------

    @property
    def name(self) -> NodeName:
        return self._name

    @property
    def kind(self) -> NodeKind:
        return self._kind

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @property
    def retry(self) -> int:
        return self._retry

    @property
    def timeout(self) -> Optional[float]:
        return self._timeout

    # -- invocation ---------------------------------------------------------

    def invoke(self, state: StateT, **kwargs: Any) -> StateUpdate:
        """Invoke the node synchronously.

        Raises ``TypeError`` if the underlying callable is async.
        """
        if self._kind == NodeKind.ASYNC:
            raise TypeError(
                f"Node '{self._name}' is async; use ``ainvoke()`` instead."
            )
        if self._kind == NodeKind.SUBGRAPH:
            return self._run_subgraph(state)
        if self._kind == NodeKind.PIPELINE:
            return self._run_pipeline(state)
        logger.debug("Node.invoke(%r)", self._name)
        return self._fn(state, **kwargs)  # type: ignore[call-arg]

    async def ainvoke(self, state: StateT, **kwargs: Any) -> StateUpdate:
        """Invoke the node asynchronously."""
        if self._kind == NodeKind.FUNCTION:
            return self._fn(state, **kwargs)  # type: ignore[call-arg]
        if self._kind == NodeKind.ASYNC:
            return await self._fn(state, **kwargs)  # type: ignore[call-arg]
        if self._kind == NodeKind.SUBGRAPH:
            return await self._arun_subgraph(state)
        if self._kind == NodeKind.PIPELINE:
            return await self._arun_pipeline(state)
        raise TypeError(
            f"Node '{self._name}' is neither sync nor async."
        )

    def stream(self, state: StateT, **kwargs: Any) -> Any:
        """Invoke the node as a generator (for streaming)."""
        if self._kind == NodeKind.STREAM:
            return self._fn(state, **kwargs)  # type: ignore[call-arg]
        raise TypeError(
            f"Node '{self._name}' is not a generator node."
        )

    async def astream(self, state: StateT, **kwargs: Any) -> Any:
        """Invoke the node as an async generator."""
        if self._kind == NodeKind.ASYNC_STREAM:
            return self._fn(state, **kwargs)  # type: ignore[call-arg]
        raise TypeError(
            f"Node '{self._name}' is not an async generator node."
        )

    # -- subgraph / pipeline helpers ---------------------------------------

    def _run_subgraph(self, state: StateT) -> StateUpdate:
        from graphforge._graph import CompiledGraph

        compiled = self._fn
        if not isinstance(compiled, CompiledGraph):
            raise TypeError(f"Node '{self._name}' is not a CompiledGraph.")
        logger.debug("Node._run_subgraph(%r)", self._name)

        # Apply input_map: create subgraph state from parent fields
        if compiled.input_map:
            kw = {}
            for parent_field, sub_field in compiled.input_map.items():
                kw[sub_field] = getattr(state, parent_field, None)
            sub_type = compiled.state_type
            if sub_type is not None:
                if hasattr(sub_type, "model_validate"):
                    state = sub_type.model_validate(kw)
                else:
                    state = sub_type(**kw)
            elif hasattr(state, "apply"):
                state = state.apply(**kw)
            else:
                for k, v in kw.items():
                    setattr(state, k, v)

        # Isolate subgraph checkpoints
        config = {"thread_id": f"sg:{self._name}"}
        result = compiled.invoke(state, config=config)

        # Apply output_map: copy subgraph result fields to parent update
        if compiled.output_map:
            result_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            mapped: Dict[str, Any] = {}
            for sub_field, parent_field in compiled.output_map.items():
                mapped[parent_field] = result_dict.get(sub_field)
            return mapped

        if hasattr(result, "model_dump"):
            return result.model_dump()
        return dict(result)

    async def _arun_subgraph(self, state: StateT) -> StateUpdate:
        from graphforge._graph import CompiledGraph

        compiled = self._fn
        if not isinstance(compiled, CompiledGraph):
            raise TypeError(f"Node '{self._name}' is not a CompiledGraph.")
        logger.debug("Node._arun_subgraph(%r)", self._name)

        # Apply input_map: create subgraph state from parent fields
        if compiled.input_map:
            kw = {}
            for parent_field, sub_field in compiled.input_map.items():
                kw[sub_field] = getattr(state, parent_field, None)
            sub_type = compiled.state_type
            if sub_type is not None:
                if hasattr(sub_type, "model_validate"):
                    state = sub_type.model_validate(kw)
                else:
                    state = sub_type(**kw)
            elif hasattr(state, "apply"):
                state = state.apply(**kw)
            else:
                for k, v in kw.items():
                    setattr(state, k, v)

        # Isolate subgraph checkpoints
        config = {"thread_id": f"sg:{self._name}"}
        result = await compiled.ainvoke(state, config=config)

        # Apply output_map
        if compiled.output_map:
            result_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            mapped: Dict[str, Any] = {}
            for sub_field, parent_field in compiled.output_map.items():
                mapped[parent_field] = result_dict.get(sub_field)
            return mapped

        if hasattr(result, "model_dump"):
            return result.model_dump()
        return dict(result)

    def _run_pipeline(self, state: StateT) -> StateUpdate:
        from graphforge.pipeline import Pipeline

        pipeline = self._fn
        if not isinstance(pipeline, Pipeline):
            raise TypeError(f"Node '{self._name}' is not a Pipeline.")
        logger.debug("Node._run_pipeline(%r)", self._name)
        return pipeline.run(state)

    async def _arun_pipeline(self, state: StateT) -> StateUpdate:
        from graphforge.pipeline import Pipeline

        pipeline = self._fn
        if not isinstance(pipeline, Pipeline):
            raise TypeError(f"Node '{self._name}' is not a Pipeline.")
        logger.debug("Node._arun_pipeline(%r)", self._name)
        return await pipeline.arun(state)

    # -- equality / repr ----------------------------------------------------

    def __repr__(self) -> str:
        return f"Node(name={self._name!r}, kind={self._kind.value})"


# ---------------------------------------------------------------------------
# Classification helper
# ---------------------------------------------------------------------------


def _classify(
    fn: Any,
) -> NodeKind:
    """Determine the :class:`NodeKind` for an arbitrary callable."""
    if isinstance(fn, Node):
        return fn._kind

    from graphforge._graph import CompiledGraph
    from graphforge.pipeline import Pipeline

    if isinstance(fn, CompiledGraph):
        return NodeKind.SUBGRAPH
    if isinstance(fn, Pipeline):
        return NodeKind.PIPELINE

    if inspect.iscoroutinefunction(fn):
        return NodeKind.ASYNC
    if inspect.isasyncgenfunction(fn):
        return NodeKind.ASYNC_STREAM
    if inspect.isgeneratorfunction(fn):
        return NodeKind.STREAM

    return NodeKind.FUNCTION


__all__ = [
    "Node",
    "NodeKind",
]
