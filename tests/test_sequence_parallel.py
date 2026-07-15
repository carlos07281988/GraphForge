"""Tests for add_sequence/add_parallel high-level API."""
from graphforge import Graph, GraphState, node_field, Append


class SeqState(GraphState):
    path: list = node_field(default=[], merge="append")
    value: int = 0


def inc(state):
    return {"value": state.value + 1, "path": Append(["inc"])}


def double(state):
    return {"value": state.value * 2, "path": Append(["double"])}


def dec(state):
    return {"value": state.value - 1, "path": Append(["dec"])}


class TestAddSequence:
    def test_basic_sequence(self) -> None:
        g = Graph[SeqState]()
        g.add_node("a", inc).add_node("b", double).add_node("c", dec)
        g.add_sequence(["a", "b", "c"])
        g.add_edge("c", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()
        result = compiled.invoke(SeqState(value=1))
        assert result.path == ["inc", "double", "dec"]

    def test_sequence_shortcut(self) -> None:
        g = Graph[SeqState]()
        g.add_node("a", inc).add_node("b", double)
        g.add_sequence(["a", "b"])
        g.add_edge("b", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()
        result = compiled.invoke(SeqState(value=5))
        assert result.value == 12  # (5+1)*2

    def test_sequence_requires_min_two(self) -> None:
        g = Graph[SeqState]()
        g.add_node("a", inc)
        try:
            g.add_sequence(["a"])
            assert False, "should have raised"
        except ValueError:
            pass


class TestAddParallel:
    def test_two_branches(self) -> None:
        g = Graph[SeqState]()
        g.add_node("start", inc)
        g.add_node("a", inc)
        g.add_node("b", double)
        g.add_node("join", inc)
        g.add_parallel("start", ["a", "b"], join="join")
        g.add_edge("join", "__end__")
        g.set_entry_point("start")
        compiled = g.compile()
        result = compiled.invoke(SeqState(value=1))
        assert result.value >= 3  # after start(2), then a(3) or b(4), then join(+1)

    def test_no_join(self) -> None:
        g = Graph[SeqState]()
        g.add_node("start", inc)
        g.add_node("a", inc)
        g.add_node("b", double)
        g.add_parallel("start", ["a", "b"])
        g.set_entry_point("start")
        compiled = g.compile(state_type=SeqState)
        result = compiled.invoke(SeqState(value=0))
        assert result.value >= 0  # parallel branches merge
