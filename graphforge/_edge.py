"""Edge definitions for graph topology.

Edges define how execution flows between nodes in a graph. GraphForge
supports two kinds of edges:

1. **Direct edges** — unconditional ``source -> target`` transitions.
2. **Conditional edges** — the ``source`` node is followed by a *router*
   function that inspects the current state and picks a target at runtime.
"""

from __future__ import annotations

from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Set,
    Union,
)

from graphforge._types import (
    AsyncRouterFunc,
    NodeName,
    RouterFunc,
    StateT,
)


# ---------------------------------------------------------------------------
# Edge type discriminator
# ---------------------------------------------------------------------------


class EdgeKind(str, Enum):
    DIRECT = "direct"
    CONDITIONAL = "conditional"


# ---------------------------------------------------------------------------
# Edge — unconditional edge between two nodes
# ---------------------------------------------------------------------------


class DirectEdge(Generic[StateT]):
    """An unconditional edge from *source* to *target*.

    When the executor finishes *source*, it will always proceed to *target*.
    """

    __slots__ = ("source", "target")

    def __init__(self, source: NodeName, target: NodeName) -> None:
        assert source, "source must be a non-empty string"
        assert target, "target must be a non-empty string"
        self.source = source
        self.target = target

    def __repr__(self) -> str:
        return f"DirectEdge({self.source!r} -> {self.target!r})"


class ConditionalEdge(Generic[StateT]):
    """A conditional edge with a router function.

    After *source* finishes, the framework calls ``router(state)`` and
    dispatches to the node name returned. The return value must be a key
    in *path_map*.

    Parameters
    ----------
    source:
        The source node name.
    router:
        A callable ``(state, **kwargs) -> str`` that returns the target
        node name based on the current state.
    path_map:
        Mapping of ``router`` return values to node names. Every value the
        router can return must be a key here.
    """

    __slots__ = ("source", "router", "path_map")

    def __init__(
        self,
        source: NodeName,
        router: Union[RouterFunc, AsyncRouterFunc],
        path_map: Mapping[str, NodeName],
    ) -> None:
        assert source, "source must be a non-empty string"
        self.source = source
        self.router = router
        self.path_map = dict(path_map)

    @property
    def targets(self) -> Set[NodeName]:
        """Return the set of possible target node names."""
        return set(self.path_map.values())

    def __repr__(self) -> str:
        return (
            f"ConditionalEdge({self.source!r} -> "
            f"{dict(self.path_map)!r})"
        )


# ---------------------------------------------------------------------------
# Union type for edges stored in a graph
# ---------------------------------------------------------------------------


class FanOutEdge(Generic[StateT]):
    """A fan-out edge that spawns multiple parallel branches."""

    __slots__ = ("source", "targets", "join")

    def __init__(
        self,
        source: NodeName,
        targets: List[NodeName],
        join: Optional[NodeName] = None,
    ) -> None:
        assert source, "source must be a non-empty string"
        assert targets, "targets must be a non-empty list"
        self.source = source
        self.targets = list(targets)
        self.join = join

    def __repr__(self) -> str:
        return (
            f"FanOutEdge({self.source!r} -> "
            f"{self.targets!r}{'' if self.join is None else f' join={self.join!r}'})"
        )


AnyEdge = Union[DirectEdge[StateT], ConditionalEdge[StateT], FanOutEdge[StateT]]


class ErrorEdge(Generic[StateT]):
    """An edge taken when a node raises an exception.

    When *source* raises an unhandled exception, execution routes to
    *fallback* instead of propagating the error.

    Parameters
    ----------
    source:
        The source node name that may fail.
    fallback:
        The fallback node to execute on failure.
    """

    __slots__ = ("source", "fallback")

    def __init__(self, source: NodeName, fallback: NodeName) -> None:
        assert source, "source must be a non-empty string"
        assert fallback, "fallback must be a non-empty string"
        self.source = source
        self.fallback = fallback

    def __repr__(self) -> str:
        return f"ErrorEdge({self.source!r} -> {self.fallback!r})"


__all__ = [
    "ConditionalEdge",
    "DirectEdge",
    "EdgeKind",
    "ErrorEdge",
    "FanOutEdge",
]
