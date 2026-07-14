# Copyright 2026 GraphForge Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from graphforge._command import Command
from graphforge._checkpoint import Checkpointer, InMemoryCheckpointer
from graphforge._edge import ConditionalEdge, FanOutEdge
from graphforge._graph import CompiledGraph
from graphforge._logging import get_logger
from graphforge._node import Node, NodeKind
from graphforge._stream import EventType, StreamEvent
from graphforge._types import (
    AsyncRouterFunc,
    ConfigDict,
    END_SENTINEL,
    NodeName,
    RouterFunc,
    StateT,
    StateUpdate,
)
from graphforge.state import GraphState, _build_reducer_map, merge_state

logger = get_logger("executor")


DEFAULT_RECURSION_LIMIT = 100


class GraphExecutionPaused(Exception):
    """Raised by a node to pause execution (e.g., waiting for human input).

    When the executor catches this exception, it saves a checkpoint and
    returns the current state instead of propagating the error.
    """

    def __init__(
        self,
        message: str = "Execution paused",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.metadata = metadata or {}
        super().__init__(message)



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
        *,
        start_node: Optional[NodeName] = None,
    ) -> StateT:
        config = config or {}
        recursion_limit = config.get("recursion_limit", DEFAULT_RECURSION_LIMIT)
        thread_id = config.get("thread_id", "default")
        checkpointer = graph.checkpointer or InMemoryCheckpointer()

        state = input_state
        node_name: Optional[NodeName] = start_node or graph.entry_point
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

            # Fan-out check: execute all branches in parallel
            if node_name in graph._fanout_map:
                logger.info("Fan-out node %r (%d branches)", node_name, len(graph._fanout_map[node_name].targets))
                state = self._execute_fanout(graph, state, node_name, config, step=step)
                fanout_edge = graph._fanout_map[node_name]
                node_name = fanout_edge.join or END_SENTINEL
                step += 1
                logger.debug("Fan-out step %d: next -> %r", step, node_name)
                continue

            logger.info("Node %r (step=%d, kind=%s)", node_name, step, node.kind.value)
            logger.debug("State before %r: %s", node_name, _dump(state))
            self._callbacks.on_node_start(node_name, _dump(state))
            try:
                updates = node.invoke(state)
            except GraphExecutionPaused as pause:
                logger.info("Node %r paused execution: %s", node_name, pause.message)
                self._callbacks.on_node_error(node_name, pause)
                if checkpointer is not None:
                    key = (thread_id, node_name, step)
                    checkpointer.put(key, _dump(state), parent_key=parent_key,
                                     metadata={"_resume_node": node_name})
                self._callbacks.on_graph_end(graph.name, _dump(state))
                return state
            except Exception as exc:
                # Retry logic
                if node.retry > 0:
                    logger.info("Node %r failed (attempt 1/%d), retrying...", node_name, node.retry + 1)
                    for attempt in range(1, node.retry + 1):
                        try:
                            updates = node.invoke(state)
                            break
                        except GraphExecutionPaused:
                            raise
                        except Exception as retry_exc:
                            last_err = retry_exc
                            if attempt < node.retry:
                                logger.info("Node %r failed (attempt %d/%d), retrying...", node_name, attempt + 1, node.retry + 1)
                                continue
                            # All retries exhausted — try fallback
                            fallback_target = graph.error_map.get(node_name)
                            if fallback_target:
                                logger.warning("Node %r failed after %d attempts, falling back to %r", node_name, node.retry + 1, fallback_target)
                                self._callbacks.on_node_error(node_name, last_err)
                                updates = {"_fallback_to": fallback_target}
                                break
                            self._callbacks.on_node_error(node_name, last_err)
                            raise last_err
                else:
                    # No retry — try fallback
                    fallback_target = graph.error_map.get(node_name)
                    if fallback_target:
                        logger.warning("Node %r failed, falling back to %r", node_name, fallback_target)
                        self._callbacks.on_node_error(node_name, exc)
                        updates = {"_fallback_to": fallback_target}
                    else:
                        logger.exception("Node %r failed at step %d", node_name, step)
                        self._callbacks.on_node_error(node_name, exc)
                        raise

            # Command API: dynamic node routing
            if isinstance(updates, Command):
                if updates.update:
                    state = _apply(state, updates.update)
                node_name = updates.goto
                logger.debug("Command routing to %r", node_name)
                step += 1
                continue
            # Check for fallback routing
            if "_fallback_to" in updates:
                node_name = updates["_fallback_to"]
                logger.warning("Routing to fallback %r", node_name)
                step += 1
                continue
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
        *,
        start_node: Optional[NodeName] = None,
    ) -> Generator[StreamEvent, None, None]:
        config = config or {}
        recursion_limit = config.get("recursion_limit", DEFAULT_RECURSION_LIMIT)
        thread_id = config.get("thread_id", "default")
        checkpointer = graph.checkpointer or InMemoryCheckpointer()

        state = input_state
        node_name: Optional[NodeName] = start_node or graph.entry_point
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

            if node_name in graph._fanout_map:
                logger.info("Stream fan-out: %r -> %d branches", node_name, len(graph._fanout_map[node_name].targets))
                state = self._execute_fanout(graph, state, node_name, config, step=step)
                fanout_edge = graph._fanout_map[node_name]
                node_name = fanout_edge.join or END_SENTINEL
                step += 1
                continue

            logger.debug("Stream emitting NODE_START for %r (step=%d)", node_name, step)
            yield StreamEvent(
                EventType.NODE_START, node=node_name, data=_dump(state), step=step
            )
            try:
                updates = node.invoke(state)
            except GraphExecutionPaused as pause:
                logger.info("Stream: Node %r paused execution: %s", node_name, pause.message)
                yield StreamEvent(
                    EventType.NODE_ERROR,
                    node=node_name,
                    data={"error": str(pause)},
                    step=step,
                )
                if checkpointer is not None:
                    key = (thread_id, node_name, step)
                    checkpointer.put(key, _dump(state), parent_key=parent_key,
                                     metadata={"_resume_node": node_name})
                node_name = END_SENTINEL
                yield StreamEvent(EventType.GRAPH_END, data={"state": _dump(state)})
                return
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

    def _execute_fanout(
        self,
        graph: CompiledGraph[StateT],
        state: StateT,
        fanout_node: NodeName,
        config: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> StateT:
        """Execute all branches from a fan-out node in parallel."""
        fanout_edge = graph._fanout_map[fanout_node]
        logger.info("Fan-out %r -> %d branches (parallel)", fanout_node, len(fanout_edge.targets))

        def run_branch(target: NodeName) -> StateT:
            """Execute a single fan-out branch."""
            if hasattr(state, "model_copy"):
                branch_state = state.model_copy(deep=False)
            else:
                branch_state = state  # type: ignore[assignment]
            current: Optional[NodeName] = target
            while current is not None and current != END_SENTINEL:
                if fanout_edge.join and current == fanout_edge.join:
                    break
                node = graph.get_node(current)
                logger.debug("Fan-out branch %r step: node %r", target, current)
                updates = node.invoke(branch_state)
                branch_state = _apply(branch_state, updates)
                current = self._resolve_next(graph, current, branch_state)
            return branch_state

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(fanout_edge.targets)) as executor:
            results: List[StateT] = list(
                executor.map(run_branch, fanout_edge.targets)
            )

        merged = _merge_parallel_results(state, results)
        logger.info("Fan-out %r done, %d branches merged (parallel)", fanout_node, len(results))
        return merged

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

    def resume(
        self,
        graph: CompiledGraph[StateT],
        thread_id: str,
        state_type: Any,
        *,
        updates: Optional[StateUpdate] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> StateT:
        """Resume execution from the last checkpoint.

        Parameters
        ----------
        graph:
            The compiled graph to resume.
        thread_id:
            Thread identifier used when creating checkpoints.
        state_type:
            The state class to reconstruct state from checkpoint dict.
        updates:
            Optional state updates to apply before resuming.
        config:
            Optional runtime configuration.

        Returns
        -------
        The final state after execution completes.
        """
        checkpointer = graph.checkpointer or InMemoryCheckpointer()
        keys = checkpointer.list(thread_id)
        if not keys:
            raise ValueError(f"No checkpoint found for thread {thread_id!r}")

        last_key = keys[-1]
        checkpoint = checkpointer.get(last_key)
        if checkpoint is None:
            raise ValueError(f"Checkpoint {last_key} not found.")

        # Reconstruct state from checkpoint dict
        if hasattr(state_type, "model_validate"):
            state = state_type.model_validate(checkpoint.state)
        elif isinstance(state_type, type):
            state = state_type(**checkpoint.state)
        else:
            raise TypeError(
                f"Cannot reconstruct state from type {state_type}. "
                f"Provide a Pydantic model class or a callable."
            )

        if updates:
            if hasattr(state, "apply"):
                state = state.apply(**updates)
            else:
                for k, v in updates.items():
                    setattr(state, k, v)

        # Determine next node after the last checkpointed node
        # If paused, resume from the pause node (re-run it)
        resume_node = checkpoint.metadata.get("_resume_node")
        if resume_node:
            logger.info(
                "Resuming graph %r from paused node %r (step=%d)",
                graph.name, resume_node, last_key[2],
            )
            return self.execute(
                graph, state,
                start_node=resume_node,
                config={**(config or {}), "thread_id": thread_id},
            )

        last_node: NodeName = last_key[1]
        next_node = self._resolve_next(graph, last_node, state)

        # If already at terminal, return state unchanged
        if next_node is None or next_node == END_SENTINEL:
            logger.info(
                "Graph %r already at terminal after node %r (step=%d)",
                graph.name, last_node, last_key[2],
            )
            return state

        logger.info(
            "Resuming graph %r from node %r (step=%d)",
            graph.name, next_node, last_key[2],
        )
        return self.execute(
            graph, state,
            start_node=next_node,
            config={**(config or {}), "thread_id": thread_id},
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
        *,
        start_node: Optional[NodeName] = None,
    ) -> StateT:
        config = config or {}
        recursion_limit = config.get("recursion_limit", DEFAULT_RECURSION_LIMIT)
        thread_id = config.get("thread_id", "default")
        checkpointer = graph.checkpointer or InMemoryCheckpointer()

        state = input_state
        node_name: Optional[NodeName] = start_node or graph.entry_point
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

            if node_name in graph._fanout_map:
                logger.info("Async fan-out node %r (%d branches)", node_name, len(graph._fanout_map[node_name].targets))
                state = await self._execute_fanout(graph, state, node_name, config, step=step)
                fanout_edge = graph._fanout_map[node_name]
                node_name = fanout_edge.join or END_SENTINEL
                step += 1
                continue

            logger.info("Async node %r (step=%d)", node_name, step)
            self._callbacks.on_node_start(node_name, _dump(state))
            try:
                if node.kind == NodeKind.ASYNC:
                    updates = await node.ainvoke(state)
                else:
                    updates = node.invoke(state)
            except GraphExecutionPaused as pause:
                logger.info("Async node %r paused: %s", node_name, pause.message)
                self._callbacks.on_node_error(node_name, pause)
                if checkpointer is not None:
                    key = (thread_id, node_name, step)
                    checkpointer.put(key, _dump(state), parent_key=parent_key,
                                     metadata={"_resume_node": node_name})
                self._callbacks.on_graph_end(graph.name, _dump(state))
                return state
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

    async def _execute_fanout(
        self,
        graph: CompiledGraph[StateT],
        state: StateT,
        fanout_node: NodeName,
        config: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> StateT:
        """Execute all branches from a fan-out node asynchronously (parallel)."""
        import asyncio

        fanout_edge = graph._fanout_map[fanout_node]
        logger.info("Fan-out %r -> %d branches (async parallel)", fanout_node, len(fanout_edge.targets))

        async def run_branch(target: NodeName) -> StateT:
            if hasattr(state, "model_copy"):
                branch_state = state.model_copy(deep=False)
            else:
                branch_state = state  # type: ignore[assignment]
            current: Optional[NodeName] = target
            while current is not None and current != END_SENTINEL:
                if fanout_edge.join and current == fanout_edge.join:
                    break
                node = graph.get_node(current)
                logger.debug("Fan-out branch %r step: node %r", target, current)
                if node.kind == NodeKind.ASYNC:
                    updates = await node.ainvoke(branch_state)
                else:
                    updates = node.invoke(branch_state)
                branch_state = _apply(branch_state, updates)
                current = self._resolve_next(graph, current, branch_state)
            return branch_state

        results: List[StateT] = await asyncio.gather(
            *[run_branch(target) for target in fanout_edge.targets]
        )

        merged = _merge_parallel_results(state, results)
        logger.info("Fan-out %r done, %d branches merged (async parallel)", fanout_node, len(results))
        return merged

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


    async def resume(
        self,
        graph: CompiledGraph[StateT],
        thread_id: str,
        state_type: Any,
        *,
        updates: Optional[StateUpdate] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> StateT:
        """Resume execution from the last checkpoint (async)."""
        checkpointer = graph.checkpointer or InMemoryCheckpointer()
        keys = checkpointer.list(thread_id)
        if not keys:
            raise ValueError(f"No checkpoint found for thread {thread_id!r}")

        last_key = keys[-1]
        checkpoint = checkpointer.get(last_key)
        if checkpoint is None:
            raise ValueError(f"Checkpoint {last_key} not found.")

        if hasattr(state_type, "model_validate"):
            state = state_type.model_validate(checkpoint.state)
        elif isinstance(state_type, type):
            state = state_type(**checkpoint.state)
        else:
            raise TypeError(
                f"Cannot reconstruct state from type {state_type}."
            )

        if updates:
            if hasattr(state, "apply"):
                state = state.apply(**updates)
            else:
                for k, v in updates.items():
                    setattr(state, k, v)

        resume_node = checkpoint.metadata.get("_resume_node")
        if resume_node:
            logger.info(
                "Resuming graph %r from paused node %r (step=%d)",
                graph.name, resume_node, last_key[2],
            )
            return await self.execute(
                graph, state,
                start_node=resume_node,
                config={**(config or {}), "thread_id": thread_id},
            )

        last_node: NodeName = last_key[1]
        next_node = self._resolve_next(graph, last_node, state)

        if next_node is None or next_node == END_SENTINEL:
            logger.info(
                "Graph %r already at terminal after node %r (step=%d)",
                graph.name, last_node, last_key[2],
            )
            return state

        logger.info(
            "Async resume graph %r from node %r (step=%d)",
            graph.name, next_node, last_key[2],
        )
        return await self.execute(
            graph, state,
            start_node=next_node,
            config={**(config or {}), "thread_id": thread_id},
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


def _merge_parallel_results(state: Any, results: List) -> Any:
    """Merge states from multiple parallel branches into one."""
    merged = state
    for branch_result in results:
        if hasattr(branch_result, 'model_dump') and hasattr(state, 'model_dump'):
            updates: Dict[str, Any] = {}
            for key in state.model_dump():
                old_val = getattr(state, key, None)
                new_val = getattr(branch_result, key, None)
                if old_val != new_val:
                    updates[key] = new_val
            if updates:
                merged = merged.apply(**updates)
        else:
            branch_dict = branch_result.model_dump() if hasattr(branch_result, 'model_dump') else dict(branch_result)
            if hasattr(merged, 'apply'):
                merged = merged.apply(**branch_dict)
            else:
                for k, v in branch_dict.items():
                    merged[k] = v
    return merged


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
