"""Tests for configuration system."""
from graphforge import Graph, GraphState, node_field


class ConfigState(GraphState):
    model: str = "default"
    temperature: float = 0.7
    output: str = ""


def model_node(state):
    return {"output": f"model={state.model}, temp={state.temperature}"}


class TestConfigurable:
    def test_configurable_invocation(self) -> None:
        g = Graph[ConfigState]()
        g.add_node("m", model_node)
        g.add_edge("m", "__end__")
        g.set_entry_point("m")
        compiled = g.compile(state_type=ConfigState)

        result = compiled.invoke(
            ConfigState(),
            configurable={"model": "gpt-4"},
        )
        assert "gpt-4" in result.output
        assert "0.7" in result.output

    def test_configurable_partial(self) -> None:
        g = Graph[ConfigState]()
        g.add_node("m", model_node)
        g.add_edge("m", "__end__")
        g.set_entry_point("m")
        compiled = g.compile(state_type=ConfigState)

        result = compiled.invoke(
            ConfigState(),
            configurable={"temperature": 0.9},
        )
        assert "0.9" in result.output

    def test_configurable_async(self) -> None:
        import asyncio

        g = Graph[ConfigState]()
        g.add_node("m", model_node)
        g.add_edge("m", "__end__")
        g.set_entry_point("m")
        compiled = g.compile(state_type=ConfigState)

        result = asyncio.run(
            compiled.ainvoke(
                ConfigState(),
                configurable={"model": "claude-3"},
            )
        )
        assert "claude-3" in result.output

    def test_no_configurable_defaults(self) -> None:
        g = Graph[ConfigState]()
        g.add_node("m", model_node)
        g.add_edge("m", "__end__")
        g.set_entry_point("m")
        compiled = g.compile(state_type=ConfigState)

        result = compiled.invoke(ConfigState())
        assert "default" in result.output
