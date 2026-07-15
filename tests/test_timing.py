"""Tests for TimingCallback."""
import time
from graphforge import Graph, GraphState, node_field
from graphforge._callbacks import TimingCallback, CallbackManager


class TState(GraphState):
    value: int = 0


def fast_node(state):
    return {"value": 1}


def slow_node(state):
    time.sleep(0.01)
    return {"value": 2}


class TestTimingCallback:
    def test_timing_records_nodes(self) -> None:
        g = Graph[TState]()
        g.add_node("a", fast_node).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=TState)

        timer = TimingCallback()
        cm = CallbackManager([timer])
        compiled.invoke(TState(), callbacks=cm)

        stats = timer.get_stats()
        assert "a" in stats
        assert stats["a"]["calls"] >= 1
        assert stats["a"]["duration"] >= 0

    def test_timing_reset(self) -> None:
        timer = TimingCallback()
        timer.on_graph_start("test", {})
        timer.on_node_start("n1", {})
        timer.on_node_end("n1", {})
        timer.on_graph_end("test", {})
        assert len(timer.get_stats()) >= 2
        timer.reset()
        assert len(timer.get_stats()) == 0

    def test_multiple_calls(self) -> None:
        g = Graph[TState]()
        g.add_node("a", slow_node).add_node("b", fast_node)
        g.add_edge("a", "b").add_edge("b", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(state_type=TState)

        timer = TimingCallback()
        compiled.invoke(TState(), callbacks=CallbackManager([timer]))
        stats = timer.get_stats()
        assert "a" in stats
        assert "b" in stats
        assert stats["b"]["duration"] >= 0
