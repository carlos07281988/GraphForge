"""Tests for managed values in parallel branches."""
from graphforge import Graph, GraphState, node_field


class MvState(GraphState):
    counter: int = 0
    path: list = node_field(default=[], merge="append")


def inc_a(state):
    return {"counter": 1, "path": ["a"]}


def inc_b(state):
    return {"counter": 2, "path": ["b"]}


class TestManagedValues:
    def test_fanout_has_managed_values_attr(self) -> None:
        g = Graph[MvState]()
        g.add_node("start", inc_a).add_node("a", inc_a).add_node("b", inc_b)
        g.add_fanout("start", ["a", "b"], join=None)
        assert any("start" in str(e) for e in g._fanout_edges)

    def test_parallel_execution(self) -> None:
        g = Graph[MvState]()
        g.add_node("start", inc_a)
        g.add_node("a", inc_a)
        g.add_node("b", inc_b)
        g.add_fanout("start", ["a", "b"])
        g.set_entry_point("start")
        compiled = g.compile(state_type=MvState)
        result = compiled.invoke(MvState())
        # Both branches run, one after another in parallel
        assert "a" in result.path
        assert "b" in result.path
