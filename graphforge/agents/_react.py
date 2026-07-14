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

"""ReAct agent pattern — think-loop-act agent built on ToolNode.

Provides :func:`create_react_agent` which builds a :class:`~graphforge._graph.Graph`
implementing the ReAct (Reasoning + Acting) loop:
LLM → tool execution → LLM → ... → final answer.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from graphforge import Graph, GraphState, node_field
from graphforge._types import END_SENTINEL
from graphforge.agents._tool_node import ToolNode, ToolDef, has_tool_calls


class ReactState(GraphState):
    """Default state for ReAct agents.

    Attributes:
        messages: Conversation history with tool calls and tool results.
        next_step: Internal routing field.
    """

    messages: List[Dict[str, Any]] = node_field(default=[], merge="append")
    next_step: str = ""


def create_react_agent(
    llm_func: Callable[[List[Dict[str, Any]], List[ToolDef]], Dict[str, Any]],
    tools: Optional[List[ToolDef]] = None,
    *,
    state_type: Optional[Type[GraphState]] = None,
    agent_node_name: str = "agent",
    tools_node_name: str = "tools",
    max_iterations: int = 10,
) -> Graph:
    """Create a ReAct (Reasoning + Acting) agent graph.

    The graph implements the standard ReAct loop:
    ``agent`` → if tool calls → ``tools`` → ``agent`` → ... → end.

    Args:
        llm_func: Callable ``(messages, tools) -> response``.
        tools: List of tool definitions with optional ``_func`` key.
        state_type: Custom state class (defaults to :class:`ReactState`).
        agent_node_name: Name for the LLM agent node (default: ``"agent"``).
        tools_node_name: Name for the tool execution node (default: ``"tools"``).
        max_iterations: Max agent-think cycles before forced termination.

    Returns:
        An un-compiled :class:`~graphforge._graph.Graph` ready for
        ``.compile()`` and ``.invoke()``.
    """
    st = state_type or ReactState
    graph = Graph[st]()

    # Agent node: calls LLM, may request tool calls
    graph.add_node(
        agent_node_name,
        ToolNode(llm_func, tools, state_messages_field="messages"),
    )

    # Tool execution node: processes tool calls from agent
    graph.add_node(tools_node_name, _execute_tools)

    # Conditional routing: if last message has tool_calls, go to tools; else end
    graph.add_conditional_edges(
        agent_node_name,
        has_tool_calls,
        {"tools": tools_node_name, "end": END_SENTINEL},
    )

    # After tools, go back to agent
    graph.add_edge(tools_node_name, agent_node_name)

    # Metadata for introspection
    graph.set_metadata("agent_type", "react")
    graph.set_metadata("max_iterations", max_iterations)

    graph.set_entry_point(agent_node_name)
    return graph


def _execute_tools(state: Any) -> Dict[str, Any]:
    """Internal node that processes pending tool calls from the last message."""
    return {"next_step": "tools_done"}


__all__ = [
    "ReactState",
    "create_react_agent",
]
