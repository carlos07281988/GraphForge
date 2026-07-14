"""Tests for premium features: retry/fallback, subgraph I/O, agents."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from graphforge import Graph, GraphState, node_field, Append, ErrorEdge
from graphforge.agents import ToolNode, has_tool_calls, create_react_agent


# =====================================================================
# Feature 1: Retry & Fallback
# =====================================================================


class RetryState(GraphState):
    x: int = 0


class TestRetry:
    def test_node_retry_on_failure(self) -> None:
        counter: List[int] = [0]

        def flaky(state: RetryState) -> dict:
            counter[0] += 1
            if counter[0] < 3:
                raise ValueError("fail")
            return {"x": state.x + 1}

        graph = (
            Graph[RetryState]()
            .add_node("flaky", flaky, retry=3)
            .add_edge("flaky", "__end__")
            .set_entry_point("flaky")
            .compile()
        )

        result = graph.invoke(RetryState(x=0))
        assert result.x == 1
        assert counter[0] == 3

    def test_node_no_retry_raises(self) -> None:
        def fail(state: RetryState) -> dict:
            raise ValueError("boom")

        graph = (
            Graph[RetryState]()
            .add_node("bad", fail)
            .add_edge("bad", "__end__")
            .set_entry_point("bad")
            .compile()
        )

        with pytest.raises(ValueError, match="boom"):
            graph.invoke(RetryState(x=0))

    def test_error_edge_fallback(self) -> None:
        def fail(state: RetryState) -> dict:
            raise ValueError("failed")

        def fallback(state: RetryState) -> dict:
            return {"x": -1}

        graph = (
            Graph[RetryState]()
            .add_node("primary", fail)
            .add_node("fallback", fallback)
            .add_error_edge("primary", "fallback")
            .add_edge("fallback", "__end__")
            .add_edge("primary", "__end__")
            .set_entry_point("primary")
            .compile()
        )

        result = graph.invoke(RetryState(x=42))
        assert result.x == -1

    def test_retry_then_fallback(self) -> None:
        counter: List[int] = [0]

        def always_fail(state: RetryState) -> dict:
            counter[0] += 1
            raise ValueError(f"fail #{counter[0]}")

        def safe(state: RetryState) -> dict:
            return {"x": 999}

        graph = (
            Graph[RetryState]()
            .add_node("risky", always_fail, retry=2)
            .add_node("safe", safe)
            .add_error_edge("risky", "safe")
            .add_edge("safe", "__end__")
            .add_edge("risky", "__end__")
            .set_entry_point("risky")
            .compile()
        )

        result = graph.invoke(RetryState(x=0))
        assert result.x == 999
        assert counter[0] == 3

    def test_default_retry_zero(self) -> None:
        graph = (
            Graph[RetryState]()
            .add_node("a", lambda s: {"x": 1})
            .add_edge("a", "__end__")
            .set_entry_point("a")
            .compile()
        )
        assert graph.get_node("a").retry == 0


# =====================================================================
# Feature 2: Subgraph I/O Mapping
# =====================================================================


class ParentState(GraphState):
    query: str = ""
    result: str = ""
    extra: str = ""


class SubState(GraphState):
    prompt: str = ""
    output: str = ""


class TestSubgraphIO:
    def test_input_map(self) -> None:
        sub = (
            Graph[SubState]()
            .add_node("echo", lambda s: {"output": f"Echo: {s.prompt}"})
            .add_edge("echo", "__end__")
            .set_entry_point("echo")
            .compile(state_type=SubState, input_map={"query": "prompt"}, output_map={"output": "result"})
        )

        parent = (
            Graph[ParentState]()
            .add_node("sub", sub)
            .add_edge("sub", "__end__")
            .set_entry_point("sub")
            .compile()
        )

        result = parent.invoke(ParentState(query="hello", result=""))
        assert "Echo: hello" in result.result

    def test_output_map(self) -> None:
        sub = (
            Graph[SubState]()
            .add_node("gen", lambda s: {"output": f"Generated: {s.prompt}"})
            .add_edge("gen", "__end__")
            .set_entry_point("gen")
            .compile(state_type=SubState, input_map={"query": "prompt"}, output_map={"output": "result"})
        )

        parent = (
            Graph[ParentState]()
            .add_node("sub", sub)
            .add_edge("sub", "__end__")
            .set_entry_point("sub")
            .compile()
        )

        result = parent.invoke(ParentState(query="test", result=""))
        assert "Generated" in result.result

    def test_bidirectional_map(self) -> None:
        sub = (
            Graph[SubState]()
            .add_node("process", lambda s: {"output": f"Processed: {s.prompt}"})
            .add_edge("process", "__end__")
            .set_entry_point("process")
            .compile(
                state_type=SubState,
                input_map={"query": "prompt"},
                output_map={"output": "result"},
            )
        )

        parent = (
            Graph[ParentState]()
            .add_node("sub", sub)
            .add_edge("sub", "__end__")
            .set_entry_point("sub")
            .compile()
        )

        result = parent.invoke(ParentState(query="world", result=""))
        assert result.result == "Processed: world"


# =====================================================================
# Feature 3: Agents (ToolNode + ReAct)
# =====================================================================


class AgentState(GraphState):
    messages: List[Dict[str, Any]] = node_field(default=[], merge="append")
    next_step: str = ""


class TestToolNode:
    def test_no_tool_calls(self) -> None:
        def llm(messages, tools):
            return {"content": "hello"}

        node = ToolNode(llm, state_messages_field="messages")
        result = node(AgentState(messages=[{"role": "user", "content": "hi"}]))
        assert "messages" in result

    def test_with_tool_calls(self) -> None:
        calls: List[str] = []

        def search(query: str) -> str:
            calls.append(query)
            return f"Results: {query}"

        def llm(messages, tools):
            return {
                "content": None,
                "tool_calls": [
                    {"id": "c1", "name": "search", "arguments": {"query": "hello"}}
                ],
            }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
                "_func": search,
            }
        ]

        node = ToolNode(llm, tools=tools, state_messages_field="messages")
        node(AgentState(messages=[{"role": "user", "content": "search"}]))
        assert len(calls) == 1
        assert calls[0] == "hello"

    def test_has_tool_calls_router(self) -> None:
        state1 = AgentState(
            messages=[{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]}]
        )
        assert has_tool_calls(state1) == "tools"

        state2 = AgentState(messages=[{"role": "assistant", "content": "done"}])
        assert has_tool_calls(state2) == "end"

    def test_tool_registry(self) -> None:
        from graphforge.agents._tool_node import ToolRegistry

        registry = ToolRegistry([])
        registry.add_func("echo", lambda msg: f"echo: {msg}")
        result = registry.execute("echo", {"msg": "hello"})
        assert result == "echo: hello"

        result = registry.execute("missing", {})
        assert "not found" in result


class TestReActAgent:
    def test_create_graph(self) -> None:
        def mock_llm(messages, tools):
            return {"content": "Done"}

        graph = create_react_agent(mock_llm)
        assert graph is not None
        compiled = graph.compile(state_type=AgentState)
        result = compiled.invoke(AgentState(messages=[{"role": "user", "content": "hello"}]))
        assert hasattr(result, "messages")
        assert len(result.messages) > 0

    def test_with_tools(self) -> None:
        call_count: List[int] = [0]

        def mock_llm(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "name": "search", "arguments": {"query": "test"}}
                    ],
                }
            return {"content": "Final answer"}

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        graph = create_react_agent(mock_llm, tools=tools)
        compiled = graph.compile(state_type=AgentState)
        result = compiled.invoke(AgentState(messages=[{"role": "user", "content": "search"}]))
        assert hasattr(result, "messages")
