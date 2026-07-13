"""Execution engine for compiled graphs.

The executor is the runtime heart of GraphForge. It walks the compiled
graph's topology, invoking nodes in order, merging their state updates,
and checkpointing progress.

Two implementations are provided:

* :class:`SyncExecutor` — for synchronous graphs.
* :class:`AsyncExecutor` — for graphs containing async nodes.

Both support standard execution (``execute()``) and streaming (``stream()``).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Generator
from typing import Any, Dict, List, Optional, Sequence

from graphforge._callbacks import CallbackManager
from graphforge._checkpoint import Checkpointer, InMemoryCheckpointer
from graphforge._edge import ConditionalEdge
from graphforge._graph import CompiledGraph
from graphforge._logging import get_logger
from graphforge._node import Node, NodeKind
from graphforge._stream import EventType, StreamEvent
from graphforge._types import NodeName, StateT, StateUpdate
from graphforge.state import GraphState, _build_reducer_map, merge_state

logger = get_logger("executor")


DEFAULT_RECURSION_LIMIT = 100
END_SENTINEL = "__end__"


# ===================================================================
# SyncExecutor
# ===================================================================


class SyncExecutor:
    """Synchronous graph executor."""

    __slots__ = ("_callbacks",)

    def __init__(
        self,
        callbacks: Optional["CallbackManager"] = None,
    ) -> None:
        self._callbacks = callbacks or CallbackManager()

    def execute(
        self,
        graph: CompiledGraph[StateT],
        input_state: StateT,
        config: Optional[Dict[str, Any]] = None,
    ) -> StateT:
        config = config or {}
        recursion_limit = config.get("recursion_limit", DEFAULT_RECURSION_LIMIT)
        thread_id = config.get("thread_id", "default")
        checkpointer = graph.checkpointer or InMemoryCheckpointer()

        state = input_state
        node_name: Optional[NodeName] = graph.entry_point
        parent_key = None
        step = 0

        logger.info(
            "Graph %r starting (thread=%r, entry=%r, recursion_limit=%d)",
            graph.name, thread_id, graph.entry_point, recursion_limit,
        )
        self._callbacks.on_graph_start(graph.name, _dump(state))

        while node_name is not None and node_name != END_SENTINEL:
            if step >= recursion_limit:
                logger.warning(
                    "Recursion limit (%d) reached for %r at step %d",
                    recursion_limit, graph.name, step,
                )
                raise RecursionError(
                    f"Graph {graph.name!r} exceeded recursion limit "
                    f"({recursion_limit}) at step {step}."
                )

            if node_name == END_SENTINEL:
                break
            node = graph.get_node(node_name)

            logger.info("Node %r (step=%d, kind=%s)", node_name, step, node.kind.value)
            logger.debug("State before %r: %s", node_name, _dump(state))
            self._callbacks.on_node_start(node_name, _dump(state))
            try:
                updates = node.invoke(state)
            except Exception as exc:
                logger.exception("Node %r failed at step %d", node_name, step)
                self._callbacks.on_node_error(node_name, exc)
                raise

            logger.debug("Node %r produced updates: %s", node_name, list(updates.keys()))
            new_state = _apply(state, updates)
            logger.debug("State after %r: %s", node_name, _dump(new_state))
            self._callbacks.on_state_update(node_name, updates, _dump(new_state))
            self._callbacks.on_node_end(node_name, _dump(new_state))

            if checkpointer is not None:
                key = (thread_id, node_name, step)
                checkpointer.put(key, _dump(new_state), parent_key=parent_key)
                parent_key = key

            state = new_state
            step += 1

            node_name = self._resolve_next(graph, node_name, state)
            logger.debug("Step %d: next node -> %r", step, node_name)

        logger.info("Graph %r finished in %d steps", graph.name, step)
        self._callbacks.on_graph_end(graph.name, _dump(state))
        return state

    def stream(
        self,
        graph: CompiledGraph[StateT],
        input_state: StateT,
        config: Optional[Dict[str, Any]] = None,
    ) -> Generator[StreamEvent, None, None]:
        config = config or {}
        recursion_limit = config.get("recursion_limit", DEFAULT_RECURSION_LIMIT)
        thread_id = config.get("thread_id", "default")
        checkpointer = graph.checkpointer or InMemoryCheckpointer()

        state = input_state
        node_name: Optional[NodeName] = graph.entry_point
        parent_key = None
        step = 0

        logger.info("Graph %r streaming start", graph.name)
        yield StreamEvent(EventType.GRAPH_START, data={"graph": graph.name})

        while node_name is not None and node_name != END_SENTINEL:
            if step >= recursion_limit:
                logger.warning("Stream recursion limit hit for %r", graph.name)
                raise RecursionError(
                    f"Graph {graph.name!r} exceeded recursion limit "
                    f"({recursion_limit}) at step {step}."
                )

            node = graph.get_node(node_name)

            logger.debug("Stream emitting NODE_START for %r (step=%d)", node_name, step)
            yield StreamEvent(
                EventType.NODE_START, node=node_name, data=_dump(state), step=step
            )
            try:
                updates = node.invoke(state)
            except Exception as exc:
                logger.exception("Stream: Node %r failed at step %d", node_name, step)
                yield StreamEvent(
                    EventType.NODE_ERROR,
                    node=node_name,
                    data={"error": str(exc)},
                    step=step,
                )
                raise

            new_state = _apply(state, updates)
            yield StreamEvent(
                EventType.STATE_UPDATE,
                node=node_name,
                data={"updates": updates, "state": _dump(new_state)},
                step=step,
            )
            yield StreamEvent(
                EventType.NODE_END, node=node_name, data=_dump(new_state), step=step
            )

            if checkpointer is not None:
                key = (thread_id, node_name, step)
                checkpointer.put(key, _dump(new_state), parent_key=parent_key)
                parent_key = key

            state = new_state
            step += 1

            next_name = self._resolve_next(graph, node_name, state)
            if node_name in graph._conditionals:
                yield StreamEvent(
                    EventType.CONDITIONAL,
                    node=node_name,
                    data={"next": next_name},
                    step=step,
                )
            node_name = next_name

        logger.info("Graph %r streaming finished", graph.name)
        yield StreamEvent(EventType.GRAPH_END, data={"state": _dump(state)})

    def _resolve_next(
        self,
        graph: CompiledGraph[StateT],
        current: NodeName,
        state: StateT,
    ) -> Optional[NodeName]:
        if current in graph._conditionals:
            cond_edge: ConditionalEdge = graph._conditionals[current]
            result = cond_edge.router(state)
            logger.debug("Routing from %r -> %r", current, result)
            self._callbacks.on_conditional_edge(
                current, str(result), str(cond_edge.path_map.get(result, END_SENTINEL))
            )
            next_node: Optional[NodeName] = cond_edge.path_map.get(result)
            if next_node is None:
                raise ValueError(
                    f"Router for node {current!r} returned {result!r}, "
                    f"which is not in path_map {cond_edge.path_map}"
                )
            return next_node

        successors: Sequence[Optional[NodeName]] = graph.successors(current)
        if not successors:
            return END_SENTINEL
        if len(successors) == 1:
            return successors[0]

        raise RuntimeError(
            f"Node {current!r} has {len(successors)} direct successors "
            f"but no conditional edge. Use a conditional edge when a node "
            f"has multiple outgoing edges."
        )


# ===================================================================
# AsyncExecutor
# ===================================================================


class AsyncExecutor:
    """Asynchronous graph executor."""

    __slots__ = ("_callbacks",)

    def __init__(
        self,
        callbacks: Optional["CallbackManager"] = None,
    ) -> None:
        self._callbacks = callbacks or CallbackManager()

    async def execute(
        self,
        graph: CompiledGraph[StateT],
        input_state: StateT,
        config: Optional[Dict[str, Any]] = None,
    ) -> StateT:
        config = config or {}
        recursion_limit = config.get("recursion_limit", DEFAULT_RECURSION_LIMIT)
        thread_id = config.get("thread_id", "default")
        checkpointer = graph.checkpointer or InMemoryCheckpointer()

        state = input_state
        node_name: Optional[NodeName] = graph.entry_point
        parent_key = None
        step = 0

        logger.info(
            "Async graph %r starting (thread=%r)", graph.name, thread_id,
        )
        self._callbacks.on_graph_start(graph.name, _dump(state))

        while node_name is not None and node_name != END_SENTINEL:
            if step >= recursion_limit:
                logger.warning("Recursion limit hit for %r", graph.name)
                raise RecursionError(
                    f"Graph {graph.name!r} exceeded recursion limit "
                    f"({recursion_limit}) at step {step}."
                )

            node = graph.get_node(node_name)

            logger.info("Async node %r (step=%d)", node_name, step)
            self._callbacks.on_node_start(node_name, _dump(state))
            try:
                if node.kind == NodeKind.ASYNC:
                    updates = await node.ainvoke(state)
                else:
                    updates = node.invoke(state)
            except Exception as exc:
                logger.exception("Async node %r failed", node_name)
                self._callbacks.on_node_error(node_name, exc)
                raise

            logger.debug("Async node %r produced: %s", node_name, list(updates.keys()))
            new_state = _apply(state, updates)
            self._callbacks.on_state_update(node_name, updates, _dump(new_state))
            self._callbacks.on_node_end(node_name, _dump(new_state))

            if checkpointer is not None:
                key = (thread_id, node_name, step)
                checkpointer.put(key, _dump(new_state), parent_key=parent_key)
                parent_key = key

            state = new_state
            step += 1

            node_name = self._resolve_next(graph, node_name, state)

        logger.info("Async graph %r finished in %d steps", graph.name, step)
        self._callbacks.on_graph_end(graph.name, _dump(state))
        return state

    async def stream(
        self,
        graph: CompiledGraph[StateT],
        input_state: StateT,
        config: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        config = config or {}
        recursion_limit = config.get("recursion_limit", DEFAULT_RECURSION_LIMIT)
        thread_id = config.get("thread_id", "default")
        checkpointer = graph.checkpointer or InMemoryCheckpointer()

        state = input_state
        node_name: Optional[NodeName] = graph.entry_point
        parent_key = None
        step = 0

        logger.info("Async graph %r stream start", graph.name)
        yield StreamEvent(EventType.GRAPH_START, data={"graph": graph.name})

        while node_name is not None and node_name != END_SENTINEL:
            if step >= recursion_limit:
                raise RecursionError(
                    f"Graph {graph.name!r} exceeded recursion limit "
                    f"({recursion_limit}) at step {step}."
                )

            node = graph.get_node(node_name)

            yield StreamEvent(
                EventType.NODE_START, node=node_name, data=_dump(state), step=step
            )
            try:
                if node.kind == NodeKind.ASYNC:
                    updates = await node.ainvoke(state)
                else:
                    updates = node.invoke(state)
            except Exception as exc:
                logger.exception("Async stream: Node %r failed", node_name)
                yield StreamEvent(
                    EventType.NODE_ERROR,
                    node=node_name,
                    data={"error": str(exc)},
                    step=step,
                )
                raise

            new_state = _apply(state, updates)
            yield StreamEvent(
                EventType.STATE_UPDATE,
                node=node_name,
                data={"updates": updates, "state": _dump(new_state)},
                step=step,
            )
            yield StreamEvent(
                EventType.NODE_END, node=node_name, data=_dump(new_state), step=step
            )

            if checkpointer is not None:
                key = (thread_id, node_name, step)
                checkpointer.put(key, _dump(new_state), parent_key=parent_key)
                parent_key = key

            state = new_state
            step += 1

            next_name = self._resolve_next(graph, node_name, state)
            if node_name in graph._conditionals:
                yield StreamEvent(
                    EventType.CONDITIONAL,
                    node=node_name,
                    data={"next": next_name},
                    step=step,
                )
            node_name = next_name

        yield StreamEvent(EventType.GRAPH_END, data={"state": _dump(state)})

    def _resolve_next(
        self,
        graph: CompiledGraph[StateT],
        current: NodeName,
        state: StateT,
    ) -> Optional[NodeName]:
        if current in graph._conditionals:
            cond_edge: ConditionalEdge = graph._conditionals[current]
            result = cond_edge.router(state)
            logger.debug("Async routing from %r -> %r", current, result)
            self._callbacks.on_conditional_edge(
                current, str(result), str(cond_edge.path_map.get(result, END_SENTINEL))
            )
            next_node: Optional[NodeName] = cond_edge.path_map.get(result)
            if next_node is None:
                raise ValueError(
                    f"Router for node {current!r} returned {result!r}, "
                    f"which is not in path_map {cond_edge.path_map}"
                )
            return next_node

        successors: Sequence[Optional[NodeName]] = graph.successors(current)
        if not successors:
            return END_SENTINEL
        if len(successors) == 1:
            return successors[0]
        raise RuntimeError(
            f"Node {current!r} has {len(successors)} direct successors "
            f"but no conditional edge."
        )


# ===================================================================
# Shared helpers
# ===================================================================


def _dump(state: Any) -> Dict[str, Any]:
    if hasattr(state, "model_dump"):
        return state.model_dump()
    if isinstance(state, dict):
        return state
    return dict(state)


def _apply(state: Any, updates: StateUpdate) -> Any:
    if hasattr(state, "apply"):
        return state.apply(**updates)
    if isinstance(state, dict):
        merged = dict(state)
        merged.update(updates)
        return merged
    raise TypeError(
        f"Cannot apply updates to state of type {type(state).__name__}. "
        f"State must have an ``apply`` method or be a dict."
    )


__all__ = [
    "AsyncExecutor",
    "SyncExecutor",
]
