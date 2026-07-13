"""Tests for streaming events during graph execution."""

from __future__ import annotations

from typing import Any, Dict

from graphforge import (
    Append,
    Graph,
    GraphState,
    node_field,
)
from graphforge._stream import EventType, StreamEvent


class _TestState(GraphState):
    value: str = ""
    path: list = node_field(default=[], merge="append")


def echo(state: TestState) -> Dict[str, Any]:
    return {"value": state.value + "!", "path": Append(["echo"])}


class TestStreamEvent:
    def test_create_event(self) -> None:
        e = StreamEvent(EventType.NODE_START, node="a", data={"key": "val"}, step=1)
        assert e.type == EventType.NODE_START
        assert e.node == "a"
        assert e.data == {"key": "val"}
        assert e.step == 1

    def test_event_repr(self) -> None:
        e = StreamEvent(EventType.NODE_START, node="a")
        r = repr(e)
        assert "node_start" in r
        assert "node='a'" in r


class TestGraphStreaming:
    def test_stream_simple(self) -> None:
        g = Graph[_TestState]()
        g.add_node("a", echo)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()

        events = list(compiled.stream(_TestState(value="x")))

        types = [e.type for e in events]
        assert EventType.GRAPH_START in types
        assert EventType.NODE_START in types
        assert EventType.STATE_UPDATE in types
        assert EventType.NODE_END in types
        assert EventType.GRAPH_END in types

        # Find the final state
        end_event = next(e for e in events if e.type == EventType.GRAPH_END)
        assert end_event.data["state"]["value"] == "x!"

    def test_stream_multiple_nodes(self) -> None:
        def first(state: TestState) -> Dict[str, Any]:
            return {"value": "first", "path": Append(["first"])}

        def second(state: TestState) -> Dict[str, Any]:
            return {"value": state.value + "_second", "path": Append(["second"])}

        g = Graph[_TestState]()
        g.add_node("a", first)
        g.add_node("b", second)
        g.add_edge("a", "b")
        g.add_edge("b", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()

        events = list(compiled.stream(_TestState()))

        node_starts = [e for e in events if e.type == EventType.NODE_START]
        assert len(node_starts) == 2
        assert node_starts[0].node == "a"
        assert node_starts[1].node == "b"
