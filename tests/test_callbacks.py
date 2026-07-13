"""Tests for the callback system."""

from __future__ import annotations

from typing import Any, Dict, List

from graphforge import Callback, CallbackManager
from graphforge._stream import EventType


class TrackingCallback:
    """A test callback that records all method calls."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def on_graph_start(self, graph_name: str, input_state: Dict[str, Any]) -> None:
        self.calls.append(f"graph_start:{graph_name}")

    def on_graph_end(self, graph_name: str, final_state: Dict[str, Any]) -> None:
        self.calls.append(f"graph_end:{graph_name}")

    def on_graph_error(self, graph_name: str, error: Exception) -> None:
        self.calls.append(f"graph_error:{graph_name}:{error}")

    def on_node_start(self, node: str, state: Dict[str, Any]) -> None:
        self.calls.append(f"node_start:{node}")

    def on_node_end(self, node: str, state: Dict[str, Any]) -> None:
        self.calls.append(f"node_end:{node}")

    def on_node_error(self, node: str, error: Exception) -> None:
        self.calls.append(f"node_error:{node}:{error}")

    def on_state_update(
        self, node: str, updates: Dict[str, Any], new_state: Dict[str, Any]
    ) -> None:
        self.calls.append(f"state_update:{node}:{updates}")

    def on_conditional_edge(
        self, node: str, router_result: str, target: str
    ) -> None:
        self.calls.append(f"conditional:{node}:{router_result}:{target}")


class TestCallbackManager:
    def test_dispatch_graph_start(self) -> None:
        tracker = TrackingCallback()
        mgr = CallbackManager([tracker])
        mgr.on_graph_start("test_graph", {"key": "val"})
        assert "graph_start:test_graph" in tracker.calls

    def test_dispatch_node_start(self) -> None:
        tracker = TrackingCallback()
        mgr = CallbackManager([tracker])
        mgr.on_node_start("node_a", {})
        assert "node_start:node_a" in tracker.calls

    def test_dispatch_node_end(self) -> None:
        tracker = TrackingCallback()
        mgr = CallbackManager([tracker])
        mgr.on_node_end("node_a", {})
        assert "node_end:node_a" in tracker.calls

    def test_dispatch_node_error(self) -> None:
        tracker = TrackingCallback()
        mgr = CallbackManager([tracker])
        mgr.on_node_error("node_a", ValueError("boom"))
        assert any("node_error:node_a" in c for c in tracker.calls)

    def test_dispatch_state_update(self) -> None:
        tracker = TrackingCallback()
        mgr = CallbackManager([tracker])
        mgr.on_state_update("node_a", {"count": 1}, {"count": 1})
        assert any("state_update:node_a" in c for c in tracker.calls)

    def test_dispatch_conditional_edge(self) -> None:
        tracker = TrackingCallback()
        mgr = CallbackManager([tracker])
        mgr.on_conditional_edge("node_a", "continue", "node_b")
        assert "conditional:node_a:continue:node_b" in tracker.calls

    def test_dispatch_graph_error(self) -> None:
        tracker = TrackingCallback()
        mgr = CallbackManager([tracker])
        mgr.on_graph_error("test_graph", RuntimeError("fail"))
        assert any("graph_error:test_graph" in c for c in tracker.calls)

    def test_add_remove_callback(self) -> None:
        tracker = TrackingCallback()
        mgr = CallbackManager()
        mgr.add(tracker)
        mgr.on_graph_start("g", {})
        assert len(tracker.calls) == 1
        mgr.remove(tracker)
        mgr.on_graph_end("g", {})
        assert len(tracker.calls) == 1  # no new call

    def test_empty_manager_no_error(self) -> None:
        mgr = CallbackManager()
        mgr.on_graph_start("g", {})
        mgr.on_node_start("n", {})
        mgr.on_graph_end("g", {})
        # Should not raise
