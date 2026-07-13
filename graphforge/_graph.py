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

from graphforge._edge import AnyEdge, ConditionalEdge, DirectEdge
from graphforge._logging import get_logger
from graphforge._node import Node, NodeKind
from graphforge._types import (
    AsyncNodeFunc,
    AsyncRouterFunc,
    AsyncStreamingNodeFunc,
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
        "_entry_point",
        "_finish_points",
        "_metadata",
    )

    def __init__(self) -> None:
        self._nodes: Dict[NodeName, Node[StateT]] = {}
        self._direct_edges: List[DirectEdge[StateT]] = []
        self._conditional_edges: List[ConditionalEdge[StateT]] = []
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Graph[StateT]:
        if name in self._nodes:
            raise ValueError(
                f"A node named {name!r} is already registered."
            )
        if isinstance(fn, Node):
            node = fn
        else:
            node = Node[StateT](name=name, fn=fn, metadata=metadata)
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

    # -- compilation ------------------------------------------------------

    def compile(
        self,
        *,
        checkpointer: Optional["Checkpointer"] = None,
        name: Optional[str] = None,
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
            entry_point=self._entry_point,
            finish_points=set(self._finish_points),
            checkpointer=checkpointer,
            name=name or "unnamed",
            metadata=dict(self._metadata),
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
            if edge.target not in registered and edge.target != "__end__":
                raise ValueError(
                    f"Edge target {edge.target!r} is not registered."
                )

        for edge in self._conditional_edges:
            if edge.source not in registered:
                raise ValueError(
                    f"Conditional edge source {edge.source!r} is not registered."
                )
            unknown = set(edge.path_map.values()) - registered - {"__end__"}
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
        "_entry_point", "_finish_points", "_checkpointer",
        "_name", "_metadata",
        "_successors", "_conditionals",
    )

    def __init__(
        self,
        *,
        nodes: Dict[NodeName, Node[StateT]],
        direct_edges: List[DirectEdge[StateT]],
        conditional_edges: List[ConditionalEdge[StateT]],
        entry_point: NodeName,
        finish_points: Set[NodeName],
        checkpointer: Optional["Checkpointer"] = None,
        name: str = "unnamed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._nodes = nodes
        self._direct_edges = direct_edges
        self._conditional_edges = conditional_edges
        self._entry_point = entry_point
        self._finish_points = finish_points
        self._checkpointer = checkpointer
        self._name = name
        self._metadata = metadata or {}

        self._successors: Dict[NodeName, List[Optional[NodeName]]] = {}
        self._conditionals: Dict[NodeName, ConditionalEdge] = {}

        for node_name in nodes:
            self._successors[node_name] = []

        for edge in direct_edges:
            target: Optional[NodeName] = (
                None if edge.target == "__end__" else edge.target
            )
            self._successors.setdefault(edge.source, []).append(target)

        for edge in conditional_edges:
            self._conditionals[edge.source] = edge

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
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def get_node(self, name: NodeName) -> Node[StateT]:
        if name not in self._nodes:
            raise KeyError(f"Node {name!r} not found in graph {self._name!r}.")
        return self._nodes[name]

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
