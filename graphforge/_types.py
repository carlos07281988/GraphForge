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

""""Core type definitions and protocols for GraphForge.

This module defines the foundational type system that underpins
all GraphForge constructs. Every abstraction in the framework
derives from these types.
"""

from __future__ import annotations

from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from typing_extensions import Self

# ---------------------------------------------------------------------------
# TypeVars — used throughout the framework for generic state typing
# ---------------------------------------------------------------------------

StateT = TypeVar("StateT", bound="GraphState")
"""Type variable bound to GraphState for generic node/edge/graph definitions."""

FieldT = TypeVar("FieldT")
"""Type variable for a single state field."""

# ---------------------------------------------------------------------------
# Primitive aliases
# ---------------------------------------------------------------------------

NodeName = str
"""Unique identifier for a graph node."""

StateUpdate = dict[str, Any]
"""A partial state update returned by a node — keys present will be merged."""

ConfigDict = dict[str, Any]
"""Runtime configuration dictionary (thread id, recursion limit, etc.)."""

# ---------------------------------------------------------------------------
# Callable type aliases for node / router functions
# ---------------------------------------------------------------------------

#: A synchronous node function: ``(state, **kwargs) -> StateUpdate``
NodeFunc = Callable[..., StateUpdate]

#: An asynchronous node function.
AsyncNodeFunc = Callable[..., Awaitable[StateUpdate]]

#: A synchronous streaming node — yields state updates.
StreamingNodeFunc = Callable[..., Generator[StateUpdate, None, None]]

#: An asynchronous streaming node.
AsyncStreamingNodeFunc = Callable[..., AsyncGenerator[StateUpdate, None]]

#: A router maps the current state to the name of the next node.
RouterFunc = Callable[..., NodeName]

#: An async router.
AsyncRouterFunc = Callable[..., Awaitable[NodeName]]

# ---------------------------------------------------------------------------
# Sentinel values
# ---------------------------------------------------------------------------

#: Sentinel string used to indicate graph exit (the "end" pseudo-node).
END_SENTINEL: str = "__end__"

# ---------------------------------------------------------------------------
# Runtime-checkable Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphState(Protocol):
    """Protocol satisfied by any state class usable in a graph.

    The framework's :class:`graphforge.state.GraphState` base model fulfills
    this protocol. User-defined classes that implement ``model_dump`` and
    ``model_copy`` (the Pydantic-shaped interface) also satisfy it.
    """

    def model_dump(self) -> dict[str, Any]:
        """Return the state as a plain dictionary."""
        ...

    def model_copy(
        self,
        *,
        update: Union[dict[str, Any], None] = None,
        deep: bool = False,
    ) -> Self:
        """Return a copy of the state, optionally with field updates."""
        ...


__all__: list[str] = [
    # TypeVars
    "StateT",
    "FieldT",
    # Type aliases
    "NodeName",
    "StateUpdate",
    "ConfigDict",
    "END_SENTINEL",
    # Callable aliases
    "NodeFunc",
    "AsyncNodeFunc",
    "StreamingNodeFunc",
    "AsyncStreamingNodeFunc",
    "RouterFunc",
    "AsyncRouterFunc",
    # Protocols
    "GraphState",
]
