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

"""Streaming event types for graph execution.

When a graph is executed with ``stream()`` or ``astream()``, it yields a
sequence of :class:`StreamEvent` instances. Each event represents a
discrete step in the execution lifecycle.

This gives callers fine-grained visibility into the graph's progress
without requiring callbacks.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """Categories of stream events."""

    NODE_START = "node_start"
    """A node is about to execute."""

    NODE_END = "node_end"
    """A node completed successfully."""

    NODE_ERROR = "node_error"
    """A node raised an exception."""

    STATE_UPDATE = "state_update"
    """The state was updated after a node."""

    CONDITIONAL = "conditional"
    """A conditional edge was evaluated."""

    GRAPH_START = "graph_start"
    """Graph execution started."""

    GRAPH_END = "graph_end"
    """Graph execution finished."""


class StreamEvent:
    """A single event emitted during graph execution.

    Parameters
    ----------
    type:
        Event category.
    node:
        Node name (may be empty for graph-level events).
    data:
        Event payload.
    step:
        Sequential step counter.
    parent:
        Optional parent node name when inside a subgraph.
    metadata:
        Optional additional metadata.
    """

    __slots__ = ("type", "node", "data", "step", "parent", "metadata")

    def __init__(
        self,
        type: EventType,
        *,
        node: str = "",
        data: Optional[Dict[str, Any]] = None,
        step: int = 0,
        parent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.type = type
        self.node = node
        self.data = data or {}
        self.step = step
        self.parent = parent
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        parent_str = f" (parent={self.parent})" if self.parent else ""
        return (
            f"StreamEvent(type={self.type.value}, "
            f"node={self.node!r}, "
            f"step={self.step}){parent_str}"
        )


__all__ = [
    "EventType",
    "StreamEvent",
]
