"""Tests for graph definition and execution."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from graphforge import (
    Append,
    CompiledGraph,
    Graph,
    GraphState,
    Node,
    node_field,
)


class CounterState(GraphState):
    count: int = 0
    path: list = node_field(default=[], merge="append")
    finished: bool = False


def increment(state: CounterState) -> Dict[str, Any]:
    return {"count": state.count + 1, "path": Append(["increment"])}


def decrement(state: CounterState) -> Dict[str, Any]:
    return {"count": state.count - 1, "path": Append(["decrement"])}


def should_continue(state: CounterState) -> str:
    return "continue" if state.count < 3 else "done"


def finalize(state: CounterState) -> Dict[str, Any]:
    return {"finished": True, "path": Append(["finalize"])}


def double_fn(state: CounterState) -> Dict[str, Any]:
    return {"count": state.count * 2, "path": Append(["double"])}


class TestGraphBuilder:
    def test_empty_graph_fails_no_entry(self) -> None:
        g = Graph[CounterState]()
        with pytest.raises(ValueError, match="entry point"):
            g.compile()

    def test_add_node(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        assert "a" in g._nodes

    def test_add_duplicate_node_raises(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        with pytest.raises(ValueError, match="already registered"):
            g.add_node("a", increment)

    def test_add_edge(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_node("b", decrement)
        g.add_edge("a", "b")
        g.add_edge("b", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()
        assert "a" in compiled.nodes
        assert "b" in compiled.nodes

    def test_validation_dangling_source(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("b", "a")
        g.set_entry_point("a")
        with pytest.raises(ValueError, match="not registered"):
            g.compile()

    def test_validation_dangling_target(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "b")
        g.set_entry_point("a")
        with pytest.raises(ValueError, match="not registered"):
            g.compile()

    def test_conditional_edges(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_node("b", decrement)
        g.add_node("end", finalize)
        g.add_conditional_edges(
            "a",
            router=should_continue,
            path_map={"continue": "b", "done": "end"},
        )
        g.add_edge("b", "a")
        g.add_edge("end", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()
        assert compiled.entry_point == "a"

    def test_set_finish_point(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        g.set_finish_point("__end__")
        compiled = g.compile()
        assert "__end__" in compiled.finish_points

    def test_is_async_detection(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()
        assert not compiled.is_async()


class TestGraphExecution:
    def test_simple_invoke(self) -> None:
        g = Graph[CounterState]()
        g.add_node("increment", increment)
        g.add_edge("increment", "__end__")
        g.set_entry_point("increment")
        compiled = g.compile()

        result = compiled.invoke(CounterState(count=0))
        assert result.count == 1

    def test_multi_step(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_node("b", double_fn)
        g.add_edge("a", "b")
        g.add_edge("b", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()

        result = compiled.invoke(CounterState(count=2))
        assert result.count == 6

    def test_conditional_loop(self) -> None:
        g = Graph[CounterState]()
        g.add_node("increment", increment)
        g.add_node("finalize", finalize)
        g.add_conditional_edges(
            "increment",
            router=should_continue,
            path_map={"continue": "increment", "done": "finalize"},
        )
        g.add_edge("finalize", "__end__")
        g.set_entry_point("increment")
        compiled = g.compile()

        result = compiled.invoke(CounterState(count=0))
        assert result.finished is True
        assert result.count == 3

    def test_no_outgoing_edge(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.set_entry_point("a")
        compiled = g.compile()
        result = compiled.invoke(CounterState(count=0))
        assert result.count == 1

    def test_recursion_limit(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "a")
        g.set_entry_point("a")
        compiled = g.compile()

        with pytest.raises(RecursionError, match="recursion limit"):
            compiled.invoke(CounterState(count=0), config={"recursion_limit": 5})

    def test_nodes_property(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()
        assert "a" in compiled.nodes
        assert isinstance(compiled.nodes["a"], Node)

    def test_get_node(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()
        node = compiled.get_node("a")
        assert node.name == "a"

    def test_get_node_missing(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()
        with pytest.raises(KeyError):
            compiled.get_node("nonexistent")

    def test_successors(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_node("b", decrement)
        g.add_edge("a", "b")
        g.add_edge("b", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()
        succ_a = compiled.successors("a")
        assert succ_a == ["b"]
        succ_b = compiled.successors("b")
        assert succ_b == [None]


class TestCompiledGraph:
    def test_metadata(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        g.set_metadata("version", "1.0")
        compiled = g.compile(name="test_graph")
        assert compiled.name == "test_graph"
        assert compiled.metadata["version"] == "1.0"

    def test_repr(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()
        r = repr(compiled)
        assert "CompiledGraph" in r
        assert "nodes=1" in r
        assert "edges=1" in r
