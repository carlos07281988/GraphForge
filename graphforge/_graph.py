# Copyright 2024 GraphForge Contributors
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

"""Graph builder and compiled graph.

This module contains the two core graph abstractions:

* :class:`Graph` — a mutable builder that lets you declare nodes and edges.
* :class:`CompiledGraph` — the immutable, compiled graph produced by
  :meth:`Graph.compile`. This is what you ``invoke(...)`` at runtime.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import AsyncGenerator, Generator
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union,
)

from graphforge._edge import AnyEdge, ConditionalEdge, DirectEdge, FanOutEdge
from graphforge._logging import get_logger
from graphforge._node import Node, NodeKind
from graphforge._types import (
    AsyncNodeFunc,
    AsyncRouterFunc,
    AsyncStreamingNodeFunc,
    ConfigDict,
    END_SENTINEL,
    NodeFunc,
    NodeName,
    RouterFunc,
    StateT,
    StateUpdate,
    StreamingNodeFunc,
)

if TYPE_CHECKING:
    from graphforge._callbacks import CallbackManager
    from graphforge._checkpoint import Checkpointer
    from graphforge._executor import SyncExecutor
    from graphforge.pipeline import Pipeline
    from graphforge.state import GraphState

logger = get_logger("graph")


# ── Placeholder for deserialised nodes ────────────────────────────────


def _placeholder_fn(state: Any) -> Dict[str, Any]:
    """Placeholder for deserialised nodes that need a real function."""
    raise RuntimeError(
        "This node is a placeholder from deserialize(). "
        "Replace it with a real function using graph.add_node(name, fn)."
    )


# ===================================================================
# Graph — mutable builder
# ===================================================================


class Graph(Generic[StateT]):
    """A mutable, type-safe builder for execution graphs.

    Use the fluent API to declare nodes and edges, then call
    :meth:`compile` to produce an immutable :class:`CompiledGraph`.
    """

    __slots__ = (
        "_nodes",
        "_direct_edges",
        "_conditional_edges",
        "_fanout_edges",
        "_error_map",
        "_entry_point",
        "_finish_points",
        "_metadata",
    )

    def __init__(self) -> None:
        self._nodes: Dict[NodeName, Node[StateT]] = {}
        self._direct_edges: List[DirectEdge[StateT]] = []
        self._conditional_edges: List[ConditionalEdge[StateT]] = []
        self._fanout_edges: List[FanOutEdge[StateT]] = []
        self._error_map: Dict[NodeName, NodeName] = {}
        self._entry_point: Optional[NodeName] = None
        self._finish_points: Set[NodeName] = set()
        self._metadata: Dict[str, Any] = {}

    # -- node registration ------------------------------------------------

    def add_node(
        self,
        name: NodeName,
        fn: Union[
            NodeFunc,
            AsyncNodeFunc,
            StreamingNodeFunc,
            AsyncStreamingNodeFunc,
            "CompiledGraph[StateT]",
            "Pipeline[StateT]",
            Node[StateT],
        ],
        *,
        retry: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Graph[StateT]:
        if name in self._nodes and self._nodes[name]._fn is not _placeholder_fn:
            raise ValueError(
                f"A node named {name!r} is already registered."
            )
        if isinstance(fn, Node):
            node = fn
        else:
            node = Node[StateT](name=name, fn=fn, retry=retry, timeout=timeout, metadata=metadata)
        self._nodes[name] = node
        logger.debug("add_node(%r): kind=%s", name, node.kind.value)
        return self

    # -- edge registration ------------------------------------------------

    def add_edge(self, source: NodeName, target: NodeName) -> Graph[StateT]:
        self._direct_edges.append(DirectEdge(source=source, target=target))
        logger.debug("add_edge: %r -> %r", source, target)
        return self

    def add_conditional_edges(
        self,
        source: NodeName,
        router: Union[RouterFunc, AsyncRouterFunc],
        path_map: Mapping[str, NodeName],
    ) -> Graph[StateT]:
        self._conditional_edges.append(
            ConditionalEdge(source=source, router=router, path_map=path_map)
        )
        logger.debug("add_conditional_edges: %r -> %s", source, dict(path_map))
        return self

    def add_fanout(
        self,
        source: NodeName,
        targets: List[NodeName],
        join: Optional[NodeName] = None,
    ) -> Graph[StateT]:
        """Fan out from *source* to multiple *targets* that execute in parallel.

        Parameters
        ----------
        source:
            Source node name.
        targets:
            List of target node names for parallel execution.
        join:
            Optional join node where all branches converge after execution.
        """
        self._fanout_edges.append(
            FanOutEdge(source=source, targets=targets, join=join)
        )
        logger.debug("add_fanout: %r -> %s (join=%r)", source, targets, join)
        return self

    def add_error_edge(
        self,
        source: NodeName,
        fallback: NodeName,
    ) -> Graph[StateT]:
        """Add an error-handling edge from *source* to *fallback*.

        If *source* raises an exception during execution, the graph will
        route to *fallback* instead of propagating the error.
        """
        self._error_map[source] = fallback
        logger.debug("add_error_edge: %r -> %r", source, fallback)
        return self

    # -- entry / finish points --------------------------------------------

    def set_entry_point(self, name: NodeName) -> Graph[StateT]:
        self._entry_point = name
        logger.debug("set_entry_point(%r)", name)
        return self

    def set_finish_point(self, name: NodeName) -> Graph[StateT]:
        self._finish_points.add(name)
        return self

    def set_metadata(self, key: str, value: Any) -> Graph[StateT]:
        self._metadata[key] = value
        return self


    # -- serialisation -------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """Serialize the graph topology (node names, edges, metadata) to a dict.

        Node function bodies are **not** serialised.  After deserialising you
        must re-register real functions with :meth:`add_node` and re-register
        conditional-edge routers with :meth:`add_conditional_edges`.
        """
        return {
            "version": "1.0",
            "node_specs": {
                name: {
                    "retry": node.retry,
                    "timeout": node.timeout,
                    "metadata": node.metadata,
                }
                for name, node in self._nodes.items()
            },
            "direct_edges": [
                {"source": e.source, "target": e.target}
                for e in self._direct_edges
            ],
            "conditional_edges": [
                {"source": e.source, "paths": dict(e.path_map)}
                for e in self._conditional_edges
            ],
            "fanout_edges": [
                {"source": e.source, "targets": list(e.targets), "join": e.join}
                for e in self._fanout_edges
            ],
            "error_edges": [
                {"source": source, "fallback": fallback}
                for source, fallback in self._error_map.items()
            ],
            "entry_point": self._entry_point,
            "finish_points": list(self._finish_points),
            "metadata": dict(self._metadata),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "Graph[StateT]":
        """Create a :class:`Graph` builder from serialised topology data.

        Node functions and conditional-edge routers are **not** restored ---
        replace them with :meth:`add_node` and :meth:`add_conditional_edges`
        before calling :meth:`compile`.

        Parameters
        ----------
        data:
            A dict previously returned by :meth:`serialize`.

        Returns
        -------
        A :class:`Graph` builder with placeholder node functions.
        """
        graph: Graph[StateT] = cls()

        for name, spec in data.get("node_specs", {}).items():
            graph.add_node(
                name,
                _placeholder_fn,
                retry=spec.get("retry", 0),
                timeout=spec.get("timeout"),
                metadata=spec.get("metadata"),
            )

        for e in data.get("direct_edges", []):
            graph.add_edge(e["source"], e["target"])

        for e in data.get("conditional_edges", []):
            graph.add_conditional_edges(
                e["source"], _placeholder_fn, e["paths"],
            )

        for e in data.get("fanout_edges", []):
            graph.add_fanout(
                e["source"], e["targets"], join=e.get("join"),
            )

        for e in data.get("error_edges", []):
            graph.add_error_edge(e["source"], e["fallback"])

        if data.get("entry_point"):
            graph.set_entry_point(data["entry_point"])

        for fp in data.get("finish_points", []):
            graph.set_finish_point(fp)

        for k, v in data.get("metadata", {}).items():
            graph.set_metadata(k, v)

        return graph

    # -- compilation ------------------------------------------------------

    def compile(
        self,
        *,
        input_map: Optional[Dict[str, str]] = None,
        output_map: Optional[Dict[str, str]] = None,
        checkpointer: Optional["Checkpointer"] = None,
        name: Optional[str] = None,
        state_type: Optional[type] = None,
    ) -> "CompiledGraph[StateT]":
        logger.info(
            "Compiling graph: %d nodes, %d edges, %d conditional edges",
            len(self._nodes), len(self._direct_edges), len(self._conditional_edges),
        )
        self._validate()
        return CompiledGraph[StateT](
            nodes=dict(self._nodes),
            direct_edges=list(self._direct_edges),
            conditional_edges=list(self._conditional_edges),
            fanout_edges=list(self._fanout_edges),
            error_edges=dict(self._error_map),
            entry_point=self._entry_point,
            finish_points=set(self._finish_points),
            input_map=input_map,
            output_map=output_map,
            checkpointer=checkpointer,
            name=name or "unnamed",
            metadata=dict(self._metadata),
            state_type=state_type,
        )

    def _validate(self) -> None:
        logger.debug("Validating graph topology...")
        if not self._entry_point:
            raise ValueError("Graph must have an entry point.")
        if self._entry_point not in self._nodes:
            raise ValueError(
                f"Entry point {self._entry_point!r} is not a registered node. "
                f"Registered: {list(self._nodes)}"
            )

        registered = set(self._nodes)
        for edge in self._direct_edges:
            if edge.source not in registered:
                raise ValueError(
                    f"Edge source {edge.source!r} is not registered."
                )
            if edge.target not in registered and edge.target != END_SENTINEL:
                raise ValueError(
                    f"Edge target {edge.target!r} is not registered."
                )

        for edge in self._fanout_edges:
            if edge.source not in registered:
                raise ValueError(
                    f"Fan-out source {edge.source!r} is not registered."
                )
            for t in edge.targets:
                if t not in registered and t != END_SENTINEL:
                    raise ValueError(
                        f"Fan-out target {t!r} is not registered."
                    )

        for edge in self._conditional_edges:
            if edge.source not in registered:
                raise ValueError(
                    f"Conditional edge source {edge.source!r} is not registered."
                )
            unknown = set(edge.path_map.values()) - registered - {END_SENTINEL}
            if unknown:
                raise ValueError(
                    f"Conditional edge targets {unknown} are not registered."
                )


# ===================================================================
# CompiledGraph — immutable executable graph
# ===================================================================


class CompiledGraph(Generic[StateT]):
    """An immutable, compiled graph ready for execution."""

    __slots__ = (
        "_nodes", "_direct_edges", "_conditional_edges",
        "_fanout_edges",
        "_entry_point", "_finish_points", "_checkpointer",
        "_name", "_metadata",
        "_state_type",
        "_successors", "_conditionals", "_fanout_map",
        "_error_map",
        "_input_map",
        "_output_map",
    )

    def __init__(
        self,
        *,
        nodes: Dict[NodeName, Node[StateT]],
        direct_edges: List[DirectEdge[StateT]],
        conditional_edges: List[ConditionalEdge[StateT]],
        fanout_edges: Optional[List[FanOutEdge[StateT]]] = None,
        error_edges: Optional[Dict[NodeName, NodeName]] = None,
        input_map: Optional[Dict[str, str]] = None,
        output_map: Optional[Dict[str, str]] = None,
        entry_point: NodeName,
        finish_points: Set[NodeName],
        checkpointer: Optional["Checkpointer"] = None,
        name: str = "unnamed",
        metadata: Optional[Dict[str, Any]] = None,
        state_type: Optional[type] = None,
    ) -> None:
        self._nodes = nodes
        self._direct_edges = direct_edges
        self._conditional_edges = conditional_edges
        self._entry_point = entry_point
        self._finish_points = finish_points
        self._checkpointer = checkpointer
        self._name = name
        self._input_map = input_map or {}
        self._output_map = output_map or {}
        self._metadata = metadata or {}
        self._state_type = state_type

        self._successors: Dict[NodeName, List[Optional[NodeName]]] = {}
        self._conditionals: Dict[NodeName, ConditionalEdge] = {}

        for node_name in nodes:
            self._successors[node_name] = []

        for edge in direct_edges:
            target: Optional[NodeName] = (
                None if edge.target == END_SENTINEL else edge.target
            )
            self._successors.setdefault(edge.source, []).append(target)

        for edge in conditional_edges:
            self._conditionals[edge.source] = edge

        # Build fan-out lookup
        self._fanout_map: Dict[NodeName, FanOutEdge[StateT]] = {}
        for edge in (fanout_edges or []):
            self._fanout_map[edge.source] = edge

        # Build error-edge lookup
        self._error_map: Dict[NodeName, NodeName] = {}
        self._error_map.update(error_edges or {})

        logger.debug(
            "CompiledGraph %r: %d nodes, %d direct edges, %d conditional edges",
            name, len(nodes), len(direct_edges), len(conditional_edges),
        )

    # -- read-only properties ---------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> Dict[NodeName, Node[StateT]]:
        return dict(self._nodes)

    @property
    def entry_point(self) -> NodeName:
        return self._entry_point

    @property
    def finish_points(self) -> Set[NodeName]:
        return set(self._finish_points)

    @property
    def checkpointer(self) -> Optional["Checkpointer"]:
        return self._checkpointer

    @property
    def error_map(self) -> Dict[NodeName, NodeName]:
        return dict(self._error_map)

    @property
    def input_map(self) -> Dict[str, str]:
        return dict(self._input_map)

    @property
    def output_map(self) -> Dict[str, str]:
        return dict(self._output_map)

    @property
    def state_type(self) -> Optional[type]:
        """The state class used for checkpoint deserialization."""
        return self._state_type

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def get_node(self, name: NodeName) -> Node[StateT]:
        if name not in self._nodes:
            raise KeyError(f"Node {name!r} not found in graph {self._name!r}.")
        return self._nodes[name]

    def replace_node(
        self,
        name: NodeName,
        fn: Any,
        *,
        retry: int = 0,
        timeout: Optional[float] = None,
    ) -> None:
        """Replace a node's function at runtime without recompiling."""
        if name not in self._nodes:
            raise KeyError(f"Node {name!r} not found in graph {self._name!r}.")
        old_node = self._nodes[name]
        new_node = Node[StateT](
            name=name, fn=fn, retry=retry, timeout=timeout,
            metadata=old_node.metadata,
        )
        self._nodes[name] = new_node
        logger.debug("replace_node(%r): function replaced", name)

    def successors(self, node_name: NodeName) -> Sequence[Optional[NodeName]]:
        return list(self._successors.get(node_name, []))

    def is_async(self) -> bool:
        return any(
            node.kind in (NodeKind.ASYNC, NodeKind.ASYNC_STREAM)
            for node in self._nodes.values()
        )

    # -- execution --------------------------------------------------------

    def invoke(
        self,
        input_state: StateT,
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional["CallbackManager"] = None,
    ) -> StateT:
        from graphforge._executor import SyncExecutor
        executor = SyncExecutor(callbacks=callbacks)
        return executor.execute(self, input_state, config=config)

    async def ainvoke(
        self,
        input_state: StateT,
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional["CallbackManager"] = None,
    ) -> StateT:
        from graphforge._executor import AsyncExecutor
        executor = AsyncExecutor(callbacks=callbacks)
        return await executor.execute(self, input_state, config=config)

    def stream(
        self,
        input_state: StateT,
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional["CallbackManager"] = None,
    ) -> Generator["StreamEvent", None, None]:
        from graphforge._executor import SyncExecutor
        executor = SyncExecutor(callbacks=callbacks)
        yield from executor.stream(self, input_state, config=config)

    async def astream(
        self,
        input_state: StateT,
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional["CallbackManager"] = None,
    ) -> AsyncGenerator["StreamEvent", None]:
        from graphforge._executor import AsyncExecutor
        executor = AsyncExecutor(callbacks=callbacks)
        async for event in executor.stream(self, input_state, config=config):
            yield event

    def resume(
        self,
        thread_id: str,
        *,
        state_type: Optional[type] = None,
        updates: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional["CallbackManager"] = None,
    ) -> StateT:
        """Resume execution from the last checkpoint.

        Uses the graph's checkpointer to load the last saved state and
        continues execution from where it left off.

        Parameters
        ----------
        thread_id:
            Thread identifier used when creating checkpoints.
        state_type:
            State class for deserialization. Falls back to the compiled
            graph's stored ``state_type``.
        updates:
            Optional state updates to apply before resuming (e.g., human
            input injected into a paused agent).
        config:
            Optional runtime configuration.
        callbacks:
            Optional callback manager for lifecycle hooks.

        Returns
        -------
        The final state after execution completes.
        """
        from graphforge._executor import SyncExecutor

        st = state_type or self._state_type
        if st is None:
            raise ValueError(
                "No state_type provided. Either set it in compile() "
                "or pass it to resume()."
            )
        executor = SyncExecutor(callbacks=callbacks)
        return executor.resume(
            self, thread_id, st,
            updates=updates, config=config,
        )

    async def aresume(
        self,
        thread_id: str,
        *,
        state_type: Optional[type] = None,
        updates: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        callbacks: Optional["CallbackManager"] = None,
    ) -> StateT:
        """Resume execution from the last checkpoint (async)."""
        from graphforge._executor import AsyncExecutor

        st = state_type or self._state_type
        if st is None:
            raise ValueError(
                "No state_type provided. Either set it in compile() "
                "or pass it to resume()."
            )
        executor = AsyncExecutor(callbacks=callbacks)
        return await executor.resume(
            self, thread_id, st,
            updates=updates, config=config,
        )

    def __repr__(self) -> str:
        return (
            f"CompiledGraph(name={self._name!r}, "
            f"nodes={len(self._nodes)}, "
            f"edges={len(self._direct_edges) + len(self._conditional_edges)})"
        )


__all__ = [
    "CompiledGraph",
    "Graph",
]
