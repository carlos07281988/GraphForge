"""Tests for cancellation API."""
import threading
import time
from graphforge import Graph, GraphState


class SlowState(GraphState):
    value: int = 0


def slow_node(state):
    time.sleep(0.5)
    return {"value": state.value + 1}


class TestCancellation:
    def test_signal_and_clear(self) -> None:
        from graphforge._executor import _signal_cancel, _clear_cancel, _cancel_events
        _signal_cancel("test_thread")
        assert "test_thread" in _cancel_events
        assert _cancel_events["test_thread"].is_set()
        _clear_cancel("test_thread")
        assert "test_thread" not in _cancel_events

    def test_cancel_method_exists(self) -> None:
        g = Graph[SlowState]()
        g.add_node("a", slow_node).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=SlowState)
        assert hasattr(compiled, "cancel")
        assert hasattr(compiled, "clear_cancel")

    def test_cancel_accepts_thread_id(self) -> None:
        g = Graph[SlowState]()
        g.add_node("a", slow_node).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=SlowState)
        # Should not raise
        compiled.cancel("test_thread_id")
        compiled.clear_cancel("test_thread_id")
