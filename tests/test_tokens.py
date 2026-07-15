"""Tests for token-level streaming (generator nodes)."""
from graphforge import Graph, GraphState, node_field
from graphforge._node import NodeKind
from graphforge._stream import EventType


class TokenState(GraphState):
    output: str = ""
    tokens: list = node_field(default=[], merge="append")


def token_generator(state):
    """Generator function that yields individual tokens."""
    for ch in ["hello", " ", "world"]:
        yield {"output": ch, "tokens": [ch]}


class TestTokenStreaming:
    def test_node_classifies_as_stream(self) -> None:
        g = Graph[TokenState]()
        g.add_node("gen", token_generator)
        node = g._nodes["gen"]
        assert node.kind == NodeKind.STREAM

    def test_generator_in_execute(self) -> None:
        g = Graph[TokenState]()
        g.add_node("gen", token_generator)
        g.add_edge("gen", "__end__")
        g.set_entry_point("gen")
        compiled = g.compile(state_type=TokenState)
        result = compiled.invoke(TokenState())
        assert "world" in result.output

    def test_generator_in_stream(self) -> None:
        g = Graph[TokenState]()
        g.add_node("gen", token_generator)
        g.add_edge("gen", "__end__")
        g.set_entry_point("gen")
        compiled = g.compile(state_type=TokenState)
        events = list(compiled.stream(TokenState()))
        token_events = [e for e in events if e.type == EventType.STREAM_TOKEN]
        assert len(token_events) > 0  # at least SOME token events
        # Stream should emit more events than just the basic NODE_START/NODE_END
        assert len(events) > 2

    def test_generator_multiple_fields(self) -> None:
        """Generator yields are partial state updates (last value wins)."""
        g = Graph[TokenState]()
        g.add_node("gen", token_generator)
        g.add_edge("gen", "__end__")
        g.set_entry_point("gen")
        compiled = g.compile(state_type=TokenState)
        result = compiled.invoke(TokenState())
        # Each yield overwrites previous values, so final state has last token
        assert result.output == "world"
        assert result.tokens == ["world"]
