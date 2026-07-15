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



class TimingCallback:
    """Callback that records execution timing per node.

    Provides :meth:`get_stats` returning a dict of ``node_name -> timing_info``.

    Usage::

        from graphforge import CallbackManager
        from graphforge._callbacks import TimingCallback

        timer = TimingCallback()
        cm = CallbackManager([timer])
        graph.invoke(state, callbacks=cm)

        for node, stats in timer.get_stats().items():
            print(f"{node}: {stats['duration']:.3f}s")
    """

    def __init__(self) -> None:
        self._node_times: Dict[str, Dict[str, float]] = {}
        self._graph_start: Optional[float] = None
        self._graph_name: str = ""
        self._current_node: Optional[str] = None
        self._current_start: Optional[float] = None

    def on_graph_start(self, graph_name: str, input_state: Dict[str, Any]) -> None:
        self._graph_name = graph_name
        self._graph_start = __import__("time").time()

    def on_graph_end(self, graph_name: str, final_state: Dict[str, Any]) -> None:
        if self._graph_start is not None:
            elapsed = __import__("time").time() - self._graph_start
            self._node_times["_graph_total"] = {
                "name": graph_name,
                "duration": elapsed,
            }

    def on_node_start(self, node: str, state: Dict[str, Any]) -> None:
        self._current_node = node
        self._current_start = __import__("time").time()

    def on_node_end(self, node: str, state: Dict[str, Any]) -> None:
        if self._current_start is not None and self._current_node is not None:
            elapsed = __import__("time").time() - self._current_start
            if self._current_node not in self._node_times:
                self._node_times[self._current_node] = {
                    "name": self._current_node,
                    "duration": 0.0,
                    "calls": 0,
                }
            self._node_times[self._current_node]["duration"] += elapsed
            self._node_times[self._current_node]["calls"] =                 self._node_times[self._current_node].get("calls", 0) + 1
        self._current_node = None
        self._current_start = None

    def on_node_error(self, node: str, error: Exception) -> None:
        self._current_node = None
        self._current_start = None

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return recorded timing statistics."""
        return dict(self._node_times)

    def reset(self) -> None:
        """Clear all recorded statistics."""
        self._node_times.clear()
        self._graph_start = None
        self._graph_name = ""
        self._current_node = None
        self._current_start = None



# ---------------------------------------------------------------------------
# CostCallback — token usage and cost tracking
# ---------------------------------------------------------------------------

# Default pricing: (input_price_per_1k, output_price_per_1k) in USD
_DEFAULT_PRICING = {
    "gpt-4": (0.03, 0.06),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4o": (0.01, 0.03),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-3.5-turbo": (0.0015, 0.002),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3.5-sonnet": (0.003, 0.015),
    "default": (0.01, 0.03),
}


class CostCallback:
    """Callback that tracks token usage and estimated costs per node.

    Usage::

        from graphforge import CallbackManager
        from graphforge._callbacks import CostCallback

        cost = CostCallback()
        cm = CallbackManager([cost])
        compiled.invoke(state, callbacks=cm)

        print(f"Total cost: ${cost.total_cost():.4f}")
        for node, info in cost.get_stats().items():
            print(f"  {node}: {info['cost']:.4f}")
    """

    def __init__(self) -> None:
        self._node_costs: Dict[str, Dict[str, float]] = {}
        self._pricing: Dict[str, tuple] = dict(_DEFAULT_PRICING)
        self._current_node: Optional[str] = None

    def set_pricing(
        self, model: str, input_price: float, output_price: float
    ) -> None:
        """Set custom pricing for a model.

        Parameters
        ----------
        model:
            Model name.
        input_price:
            Price per 1K input tokens (USD).
        output_price:
            Price per 1K output tokens (USD).
        """
        self._pricing[model] = (input_price, output_price)

    def track(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        *,
        node: Optional[str] = None,
    ) -> None:
        """Record token usage for a model.

        Parameters
        ----------
        model:
            Model name (e.g. ``"gpt-4"``).
        prompt_tokens:
            Number of prompt tokens.
        completion_tokens:
            Number of completion tokens.
        node:
            Node name (defaults to current node from callbacks).
        """
        node_name = node or self._current_node or "_unknown"
        prices = self._pricing.get(model, self._pricing["default"])
        input_cost = (prompt_tokens / 1000) * prices[0]
        output_cost = (completion_tokens / 1000) * prices[1]

        if node_name not in self._node_costs:
            self._node_costs[node_name] = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0,
                "models": {},
            }

        stats = self._node_costs[node_name]
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += completion_tokens
        stats["total_tokens"] += prompt_tokens + completion_tokens
        stats["cost"] += input_cost + output_cost

        if model not in stats["models"]:
            stats["models"][model] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        stats["models"][model]["calls"] += 1
        stats["models"][model]["prompt_tokens"] += prompt_tokens
        stats["models"][model]["completion_tokens"] += completion_tokens

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return cost statistics per node."""
        return dict(self._node_costs)

    def total_cost(self) -> float:
        """Return total cost across all nodes."""
        return sum(s["cost"] for s in self._node_costs.values())

    def total_tokens(self) -> int:
        """Return total tokens across all nodes."""
        return sum(s["total_tokens"] for s in self._node_costs.values())

    def reset(self) -> None:
        """Clear all recorded statistics."""
        self._node_costs.clear()

    # Callback hooks
    def on_node_start(self, node: str, state: Dict[str, Any]) -> None:
        self._current_node = node

    def on_node_end(self, node: str, state: Dict[str, Any]) -> None:
        self._current_node = None

    def on_node_error(self, node: str, error: Exception) -> None:
        self._current_node = None


__all__ = [
    "Callback",
    "CallbackManager",
    "TimingCallback",
    "CostCallback",
]
