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

"""Lifecycle callback system.

Callbacks provide hooks into graph execution without tight coupling.
The system is inspired by LangChain's callback design but simplified:

1. Single ``Callback`` protocol instead of a class hierarchy.
2. ``CallbackManager`` as a lightweight registry.
3. Every method is optional — implement only what you need.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from graphforge._logging import get_logger

logger = get_logger("callback")


@runtime_checkable
class Callback(Protocol):
    """Protocol for lifecycle callbacks. All methods are optional."""

    def on_graph_start(self, graph_name: str, input_state: Dict[str, Any]) -> None:
        ...

    def on_graph_end(self, graph_name: str, final_state: Dict[str, Any]) -> None:
        ...

    def on_graph_error(self, graph_name: str, error: Exception) -> None:
        ...

    def on_node_start(self, node: str, state: Dict[str, Any]) -> None:
        ...

    def on_node_end(self, node: str, state: Dict[str, Any]) -> None:
        ...

    def on_node_error(self, node: str, error: Exception) -> None:
        ...

    def on_state_update(
        self, node: str, updates: Dict[str, Any], new_state: Dict[str, Any]
    ) -> None:
        ...

    def on_conditional_edge(
        self, node: str, router_result: str, target: str
    ) -> None:
        ...


class CallbackManager:
    """Manages a list of callbacks and invokes them.

    This is a **composite** — calling a method on the manager dispatches
    to every registered callback.
    """

    __slots__ = ("_callbacks",)

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self._callbacks: List[Callback] = list(callbacks or [])

    def add(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def remove(self, callback: Callback) -> None:
        self._callbacks.remove(callback)

    # -- dispatch methods -------------------------------------------------

    def on_graph_start(self, graph_name: str, input_state: Dict[str, Any]) -> None:
        logger.debug("Callback.on_graph_start(%r)", graph_name)
        for cb in self._callbacks:
            if hasattr(cb, "on_graph_start"):
                cb.on_graph_start(graph_name, input_state)

    def on_graph_end(self, graph_name: str, final_state: Dict[str, Any]) -> None:
        logger.debug("Callback.on_graph_end(%r)", graph_name)
        for cb in self._callbacks:
            if hasattr(cb, "on_graph_end"):
                cb.on_graph_end(graph_name, final_state)

    def on_graph_error(self, graph_name: str, error: Exception) -> None:
        logger.debug("Callback.on_graph_error(%r): %s", graph_name, error)
        for cb in self._callbacks:
            if hasattr(cb, "on_graph_error"):
                cb.on_graph_error(graph_name, error)

    def on_node_start(self, node: str, state: Dict[str, Any]) -> None:
        logger.debug("Callback.on_node_start(%r)", node)
        for cb in self._callbacks:
            if hasattr(cb, "on_node_start"):
                cb.on_node_start(node, state)

    def on_node_end(self, node: str, state: Dict[str, Any]) -> None:
        logger.debug("Callback.on_node_end(%r)", node)
        for cb in self._callbacks:
            if hasattr(cb, "on_node_end"):
                cb.on_node_end(node, state)

    def on_node_error(self, node: str, error: Exception) -> None:
        logger.debug("Callback.on_node_error(%r): %s", node, error)
        for cb in self._callbacks:
            if hasattr(cb, "on_node_error"):
                cb.on_node_error(node, error)

    def on_state_update(
        self, node: str, updates: Dict[str, Any], new_state: Dict[str, Any]
    ) -> None:
        logger.debug("Callback.on_state_update(%r): %s", node, list(updates.keys()))
        for cb in self._callbacks:
            if hasattr(cb, "on_state_update"):
                cb.on_state_update(node, updates, new_state)

    def on_conditional_edge(
        self, node: str, router_result: str, target: str
    ) -> None:
        logger.debug(
            "Callback.on_conditional_edge(%r): %r -> %r",
            node, router_result, target,
        )
        for cb in self._callbacks:
            if hasattr(cb, "on_conditional_edge"):
                cb.on_conditional_edge(node, router_result, target)


__all__ = [
    "Callback",
    "CallbackManager",
]
