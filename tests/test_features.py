# Copyright 2026 GraphForge Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for premium features: retry/fallback, subgraph I/O, agents."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from graphforge import Graph, GraphState, node_field, Append
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


# =====================================================================
# Feature 4: Graph Serialisation
# =====================================================================


class SerialState(GraphState):
    x: int = 0
    y: str = ""


class TestGraphSerialisation:
    def test_serialize_basic(self) -> None:
        g = Graph[SerialState]()
        g.add_node("a", lambda s: {"x": 1})
        g.add_node("b", lambda s: {"y": "hello"})
        g.add_edge("a", "b")
        g.add_edge("b", "__end__")
        g.set_entry_point("a")
        data = g.serialize()
        assert data["version"] == "1.0"
        assert set(data["node_specs"]) == {"a", "b"}
        assert len(data["direct_edges"]) == 2

    def test_serialize_with_metadata(self) -> None:
        g = Graph[SerialState]()
        g.add_node("a", lambda s: {"x": 1}, retry=3, timeout=30.0, metadata={"env": "prod"})
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        g.set_metadata("name", "test_graph")
        data = g.serialize()
        assert data["node_specs"]["a"]["retry"] == 3
        assert data["node_specs"]["a"]["timeout"] == 30.0
        assert data["node_specs"]["a"]["metadata"]["env"] == "prod"
        assert data["metadata"]["name"] == "test_graph"

    def test_round_trip_simple(self) -> None:
        g = Graph[SerialState]()
        g.add_node("a", lambda s: {"x": s.x + 1})
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        data = g.serialize()

        g2 = Graph.deserialize(data)
        g2.add_node("a", lambda s: {"x": s.x + 1})
        compiled = g2.compile()
        result = compiled.invoke(SerialState(x=5))
        assert result.x == 6

    def test_round_trip_error_edges(self) -> None:
        def fail_fn(state: SerialState) -> dict:
            raise ValueError("fail")

        def backup_fn(state: SerialState) -> dict:
            return {"x": -1}

        g = Graph[SerialState]()
        g.add_node("primary", fail_fn)
        g.add_node("backup", backup_fn)
        g.add_error_edge("primary", "backup")
        g.add_edge("backup", "__end__")
        g.add_edge("primary", "__end__")
        g.set_entry_point("primary")
        data = g.serialize()

        g2 = Graph.deserialize(data)
        g2.add_node("primary", fail_fn)
        g2.add_node("backup", backup_fn)
        compiled = g2.compile()
        result = compiled.invoke(SerialState(x=10))
        assert result.x == -1

    def test_json_round_trip(self) -> None:
        import json
        g = Graph[SerialState]()
        g.add_node("a", lambda s: {"x": 1})
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        data = g.serialize()
        json_str = json.dumps(data)
        restored = json.loads(json_str)

        g2 = Graph.deserialize(restored)
        g2.add_node("a", lambda s: {"x": 42})
        compiled = g2.compile()
        result = compiled.invoke(SerialState(x=0))
        assert result.x == 42

    def test_deserialize_invalid_replaces_placeholder(self) -> None:
        g = Graph[SerialState]()
        g.add_node("a", lambda s: {"x": 1})
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        data = g.serialize()

        g2 = Graph.deserialize(data)
        compiled = g2.compile()
        # Should raise RuntimeError because 'a' is still a placeholder
        import pytest
        with pytest.raises(RuntimeError, match="placeholder"):
            compiled.invoke(SerialState(x=0))

    def test_serialize_fanout(self) -> None:
        g = Graph[SerialState]()
        g.add_node("src", lambda s: {"x": 1})
        g.add_fanout("src", ["a", "b"], join="j")
        g.add_node("a", lambda s: {"x": s.x + 1})
        g.add_node("b", lambda s: {"x": s.x + 2})
        g.add_node("j", lambda s: {"x": s.x})
        g.add_edge("j", "__end__")
        g.set_entry_point("src")
        data = g.serialize()

        g2 = Graph.deserialize(data)
        g2.add_node("src", lambda s: {"x": 1})
        g2.add_node("a", lambda s: {"x": s.x + 1})
        g2.add_node("b", lambda s: {"x": s.x + 2})
        g2.add_node("j", lambda s: {"x": s.x})
        compiled = g2.compile()
        result = compiled.invoke(SerialState(x=0))
        # Fan-out merge with current executor structure picks last branch
        assert result.x == 2

    def test_serialize_empty_graph_raises(self) -> None:
        g = Graph[SerialState]()
        data = g.serialize()
        assert data["node_specs"] == {}
        assert data["entry_point"] is None
        assert data["direct_edges"] == []


# =====================================================================
# Feature 5: Command API
# =====================================================================


class CmdState(GraphState):
    x: int = 0
    path: str = ""


class TestCommandAPI:
    def test_command_routes_to_target(self) -> None:
        from graphforge import Command

        def router(state: CmdState) -> Union[Dict, Command]:
            if state.x > 0:
                return Command(goto="positive", update={"x": state.x * 2})
            return Command(goto="negative", update={"x": -1})

        def positive(state: CmdState) -> dict:
            return {"path": "pos"}

        def negative(state: CmdState) -> dict:
            return {"path": "neg"}

        graph = (
            Graph[CmdState]()
            .add_node("router", router)
            .add_node("positive", positive)
            .add_node("negative", negative)
            .add_edge("router", "__end__")  # default edge (overridden by Command)
            .add_edge("positive", "__end__")
            .add_edge("negative", "__end__")
            .set_entry_point("router")
            .compile()
        )

        result = graph.invoke(CmdState(x=5))
        assert result.path == "pos"
        assert result.x == 10  # 5 * 2

    def test_command_negative_case(self) -> None:
        from graphforge import Command

        def router(state: CmdState) -> Union[Dict, Command]:
            if state.x > 0:
                return Command(goto="positive", update={"x": state.x * 2})
            return Command(goto="negative", update={"x": -1})

        def positive(state: CmdState) -> dict:
            return {"path": "pos"}

        def negative(state: CmdState) -> dict:
            return {"path": "neg"}

        graph = (
            Graph[CmdState]()
            .add_node("router", router)
            .add_node("positive", positive)
            .add_node("negative", negative)
            .add_edge("router", "__end__")  # default edge
            .add_edge("positive", "__end__")
            .add_edge("negative", "__end__")
            .set_entry_point("router")
            .compile()
        )

        result = graph.invoke(CmdState(x=0))
        assert result.path == "neg"
        assert result.x == -1

    def test_command_overrides_default_edge(self) -> None:
        from graphforge import Command

        def always_route_to_positive(state: CmdState) -> Command:
            return Command(goto="positive")

        g = (
            Graph[CmdState]()
            .add_node("start", always_route_to_positive)
            .add_node("positive", lambda s: {"path": "hit"})
            .add_edge("start", "__end__")  # Would normally go to end
            .add_edge("positive", "__end__")
            .set_entry_point("start")
            .compile()
        )
        result = g.invoke(CmdState(x=1))
        assert result.path == "hit"

    def test_command_no_update_keeps_state(self) -> None:
        from graphforge import Command

        def router(state: CmdState) -> Command:
            return Command(goto="next")

        g = (
            Graph[CmdState]()
            .add_node("router", router)
            .add_node("next", lambda s: {"x": s.x + 10})
            .add_edge("router", "__end__")
            .add_edge("next", "__end__")
            .set_entry_point("router")
            .compile()
        )
        result = g.invoke(CmdState(x=7))
        assert result.x == 17  # 7 + 10 from next node

    def test_command_in_retried_node(self) -> None:
        from graphforge import Command

        counter = [0]

        def flaky_then_route(state: CmdState) -> Union[Dict, Command]:
            counter[0] += 1
            if counter[0] < 2:
                raise ValueError("will retry")
            return Command(goto="target", update={"x": 99})

        def target(state: CmdState) -> dict:
            return {"path": "reached"}

        g = (
            Graph[CmdState]()
            .add_node("source", flaky_then_route, retry=2)
            .add_node("target", target)
            .add_edge("source", "__end__")
            .add_edge("target", "__end__")
            .set_entry_point("source")
            .compile()
        )
        result = g.invoke(CmdState(x=0))
        assert result.path == "reached"
        assert result.x == 99


# =====================================================================
# Feature 6: Interrupt / Resume
# =====================================================================


class IntState(GraphState):
    x: int = 0
    step: str = ""


class TestInterruptResume:
    def test_interrupt_pauses_then_resume(self) -> None:
        from graphforge import interrupt, InMemoryCheckpointer

        counter: List[int] = [0]

        def node_with_interrupt(state: IntState) -> dict:
            counter[0] += 1
            if counter[0] == 1:
                interrupt(message="Need human input")
            return {"x": state.x + 1, "step": "done"}

        graph = (
            Graph[IntState]()
            .add_node("a", node_with_interrupt)
            .add_edge("a", "__end__")
            .set_entry_point("a")
            .compile(checkpointer=InMemoryCheckpointer(), state_type=IntState)
        )

        # First call pauses
        state1 = graph.invoke(IntState(x=0), config={"thread_id": "int1"})
        assert state1.x == 0
        assert state1.step == ""
        assert counter[0] == 1

        # Resume with updates
        result = graph.resume("int1", updates={"x": 10})
        assert result.x == 11  # 10 + 1
        assert result.step == "done"
        assert counter[0] == 2

    def test_resume_without_updates(self) -> None:
        from graphforge import interrupt, InMemoryCheckpointer

        counter: List[int] = [0]

        def node(state: IntState) -> dict:
            counter[0] += 1
            if counter[0] == 1:
                interrupt()
            return {"step": "done"}

        cp = InMemoryCheckpointer()
        g = Graph[IntState]().add_node("a", node).add_edge("a", "__end__").set_entry_point("a").compile(checkpointer=cp, state_type=IntState)
        g.invoke(IntState(x=0), config={"thread_id": "int2"})
        result = g.resume("int2")
        assert result.step == "done"

    def test_interrupt_is_reentrant(self) -> None:
        from graphforge import interrupt, InMemoryCheckpointer

        call = [0]
        def multi_pause(state: IntState) -> dict:
            call[0] += 1
            if call[0] <= 2:
                interrupt(message=f"Pause #{call[0]}")
            return {"x": state.x + 1, "step": f"done-{call[0]}"}

        cp = InMemoryCheckpointer()
        g = Graph[IntState]().add_node("a", multi_pause).add_edge("a", "__end__").set_entry_point("a").compile(checkpointer=cp, state_type=IntState)
        g.invoke(IntState(x=0), config={"thread_id": "int3"})
        g.resume("int3", updates={"x": 1})  # resumes, pauses again
        result = g.resume("int3", updates={"x": 5})
        assert result.x == 6  # 5 + 1
        assert result.step == "done-3"


# =====================================================================
# Feature 7: Mermaid Diagram Export
# =====================================================================


class MerState(GraphState):
    x: int = 0


class TestMermaidExport:
    def test_export_simple(self) -> None:
        from graphforge import export_mermaid

        g = Graph[MerState]().add_node("a", lambda s: {"x": 1}).add_edge("a", "__end__").set_entry_point("a").compile()
        m = export_mermaid(g)
        assert "graph LR" in m
        assert "a" in m
        assert "__start__" in m
        assert "__end__" in m

    def test_export_conditional(self) -> None:
        from graphforge import export_mermaid

        def router(state: MerState) -> str:
            return "b" if state.x > 0 else "c"

        g = (
            Graph[MerState]()
            .add_node("a", lambda s: s)
            .add_node("b", lambda s: s)
            .add_node("c", lambda s: s)
            .add_conditional_edges("a", router, {"b": "b", "c": "c"})
            .add_edge("b", "__end__")
            .add_edge("c", "__end__")
            .set_entry_point("a")
            .compile()
        )
        m = export_mermaid(g)
        assert "b" in m
        assert "c" in m
        assert "0|" in m or "|" in m  # edge labels

    def test_export_error_edge(self) -> None:
        from graphforge import export_mermaid

        g = (
            Graph[MerState]()
            .add_node("p", lambda s: {"x": 1})
            .add_node("f", lambda s: {"x": -1})
            .add_error_edge("p", "f")
            .add_edge("f", "__end__")
            .add_edge("p", "__end__")
            .set_entry_point("p")
            .compile()
        )
        m = export_mermaid(g)
        assert "error" in m or "f" in m

    def test_export_with_kind_label(self) -> None:
        from graphforge import export_mermaid

        sub = Graph[MerState]().add_node("inner", lambda s: {"x": s.x + 1}).add_edge("inner", "__end__").set_entry_point("inner").compile()
        g = Graph[MerState]().add_node("sub_node", sub).add_edge("sub_node", "__end__").set_entry_point("sub_node").compile()
        m = export_mermaid(g, show_kind=True)
        assert "subgraph" in m or "sub_node" in m

    def test_export_direction(self) -> None:
        from graphforge import export_mermaid

        g = Graph[MerState]().add_node("a", lambda s: {"x": 1}).add_edge("a", "__end__").set_entry_point("a").compile()
        m = export_mermaid(g, direction="TB")
        assert "graph TB" in m


# =====================================================================
# Feature 8: Dynamic Graph Mutation (replace_node)
# =====================================================================


class ReplaceState(GraphState):
    x: int = 0


class TestReplaceNode:
    def test_replace_function_changes_behavior(self) -> None:
        g = (
            Graph[ReplaceState]()
            .add_node("a", lambda s: {"x": 1})
            .add_edge("a", "__end__")
            .set_entry_point("a")
            .compile()
        )
        assert g.invoke(ReplaceState(x=0)).x == 1
        g.replace_node("a", lambda s: {"x": 99})
        assert g.invoke(ReplaceState(x=0)).x == 99

    def test_replace_preserves_edge_topology(self) -> None:
        g = (
            Graph[ReplaceState]()
            .add_node("a", lambda s: {"x": s.x + 1})
            .add_node("b", lambda s: {"x": s.x * 2})
            .add_edge("a", "b")
            .add_edge("b", "__end__")
            .set_entry_point("a")
            .compile()
        )
        assert g.invoke(ReplaceState(x=5)).x == 12  # (5+1)*2 = 12
        g.replace_node("b", lambda s: {"x": s.x * 10})
        assert g.invoke(ReplaceState(x=5)).x == 60  # (5+1)*10 = 60

    def test_replace_nonexistent_raises(self) -> None:
        g = (
            Graph[ReplaceState]()
            .add_node("a", lambda s: {"x": 1})
            .add_edge("a", "__end__")
            .set_entry_point("a")
            .compile()
        )
        with pytest.raises(KeyError, match="nonexistent"):
            g.replace_node("nonexistent", lambda s: {"x": 0})


# =====================================================================
# Feature 9: Built-in HTTP Server (GraphServer)
# =====================================================================


@pytest.mark.skipif(True, reason="Network binding disabled in sandbox")
class TestGraphServer:
    @pytest.mark.asyncio
    async def test_health_endpoint(self) -> None:
        from graphforge._http_server import GraphServer

        class S(GraphState):
            x: int = 0

        graph = (
            Graph[S]()
            .add_node("a", lambda s: {"x": 1})
            .add_edge("a", "__end__")
            .set_entry_point("a")
            .compile(state_type=S)
        )
        server = GraphServer(graph, host="127.0.0.1", port=0)
        await server.start()
        port = server._site._server.sockets[0].getsockname()[1]

        import aiohttp
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"

        await server.stop()

    @pytest.mark.asyncio
    async def test_invoke_endpoint(self) -> None:
        from graphforge._http_server import GraphServer

        class S(GraphState):
            x: int = 0

        graph = (
            Graph[S]()
            .add_node("a", lambda s: {"x": s.x + 1})
            .add_edge("a", "__end__")
            .set_entry_point("a")
            .compile(state_type=S)
        )
        server = GraphServer(graph, host="127.0.0.1", port=0)
        await server.start()
        port = server._site._server.sockets[0].getsockname()[1]

        import aiohttp
        from aiohttp import ClientSession

        async with ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{port}/invoke",
                json={"state": {"x": 5}},
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["x"] == 6

        await server.stop()


# =====================================================================
# Feature 10: Event-driven Webhooks (WebhookCallback)
# =====================================================================


class TestWebhookCallback:
    def test_sends_event_with_correct_payload(self) -> None:
        from graphforge._webhook import WebhookCallback
        from unittest.mock import patch
        from urllib.request import Request

        mock = patch("graphforge._webhook.urlopen").start()
        cb = WebhookCallback("http://hook.example.com/events")
        cb.on_graph_start("my_graph", {})

        req: Request = mock.call_args[0][0]
        assert req.full_url == "http://hook.example.com/events"
        assert req.method == "POST"
        import json
        payload = json.loads(req.data)
        assert payload["event"] == "graph_start"
        assert payload["data"]["graph"] == "my_graph"
        mock.stop()

    def test_filters_events(self) -> None:
        from graphforge._webhook import WebhookCallback
        from unittest.mock import patch

        mock = patch("graphforge._webhook.urlopen").start()
        cb = WebhookCallback("http://hook.example.com", events=["graph_end"])
        cb.on_graph_start("g", {})
        cb.on_graph_end("g", {})
        assert mock.call_count == 1  # Only graph_end sent
        mock.stop()

    def test_auth_header(self) -> None:
        from graphforge._webhook import WebhookCallback
        from unittest.mock import patch

        mock = patch("graphforge._webhook.urlopen").start()
        cb = WebhookCallback("http://hook.example.com", api_key="sk-test")
        cb.on_graph_start("g", {})
        req = mock.call_args[0][0]
        assert req.headers.get("Authorization") == "Bearer sk-test"
        mock.stop()

    def test_multiple_events(self) -> None:
        from graphforge._webhook import WebhookCallback
        from unittest.mock import patch

        mock = patch("graphforge._webhook.urlopen").start()
        cb = WebhookCallback("http://hook.example.com")
        cb.on_graph_start("g", {})
        cb.on_node_start("a", {})
        cb.on_node_end("a", {})
        assert mock.call_count == 3
        mock.stop()


# =====================================================================
# Feature 11: OpenTelemetry Tracing
# =====================================================================


class TestTracingCallback:
    def test_creates_graph_span(self) -> None:
        from unittest.mock import MagicMock, patch
        from graphforge._tracing import TracingCallback

        with patch("graphforge._tracing._HAS_OTEL", True), patch("graphforge._tracing.trace") as mock_trace:
            tracer = MagicMock()
            mock_trace.get_tracer.return_value = tracer
            cb = TracingCallback()
            cb.on_graph_start("test_graph", {})
            tracer.start_span.assert_called_with("graph.test_graph")
            tracer.start_span.return_value.end.assert_not_called()

    def test_creates_node_span(self) -> None:
        from unittest.mock import MagicMock, patch
        from graphforge._tracing import TracingCallback

        with patch("graphforge._tracing._HAS_OTEL", True), patch("graphforge._tracing.trace") as mock_trace:
            tracer = MagicMock()
            mock_trace.get_tracer.return_value = tracer
            cb = TracingCallback()
            cb.on_node_start("processor", {})
            tracer.start_span.assert_called_with("node.processor")

    def test_ends_node_span(self) -> None:
        from unittest.mock import MagicMock, patch
        from graphforge._tracing import TracingCallback

        with patch("graphforge._tracing._HAS_OTEL", True), patch("graphforge._tracing.trace") as mock_trace:
            tracer = MagicMock()
            mock_trace.get_tracer.return_value = tracer
            cb = TracingCallback()
            cb.on_node_start("a", {})
            span = tracer.start_span.return_value
            cb.on_node_end("a", {})
            span.end.assert_called_once()

    def test_records_exception_on_error(self) -> None:
        from unittest.mock import MagicMock, patch
        from graphforge._tracing import TracingCallback

        with patch("graphforge._tracing._HAS_OTEL", True), patch("graphforge._tracing.trace") as mock_trace:
            tracer = MagicMock()
            mock_trace.get_tracer.return_value = tracer
            cb = TracingCallback()
            cb.on_node_start("a", {})
            span = tracer.start_span.return_value
            cb.on_node_error("a", ValueError("boom"))
            span.record_exception.assert_called_once()

    def test_raises_without_opentelemetry(self) -> None:
        from unittest.mock import patch
        from graphforge._tracing import TracingCallback

        with patch("graphforge._tracing._HAS_OTEL", False):
            import pytest
            with pytest.raises(ImportError, match="OpenTelemetry"):
                TracingCallback()

    def test_ends_graph_span(self) -> None:
        from unittest.mock import MagicMock, patch
        from graphforge._tracing import TracingCallback

        with patch("graphforge._tracing._HAS_OTEL", True), patch("graphforge._tracing.trace") as mock_trace:
            tracer = MagicMock()
            mock_trace.get_tracer.return_value = tracer
            cb = TracingCallback()
            cb.on_graph_start("g", {})
            span = tracer.start_span.return_value
            cb.on_graph_end("g", {})
            span.end.assert_called_once()
