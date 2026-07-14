"""ToolNode — LLM tool-calling node for graph execution.

The ``ToolNode`` wraps an LLM function and a list of tool definitions.
When invoked, it calls the LLM, executes any tool calls the LLM requests,
and returns the updated message list.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from graphforge._logging import get_logger
from graphforge.state import Append

logger = get_logger("agents.tool_node")


# ── Tool definition ────────────────────────────────────────────────────

ToolFunc = Callable[..., Union[str, Dict[str, Any]]]

ToolDef = Dict[str, Any]
"""A tool definition compatible with OpenAI's tool-calling API format:

.. code-block:: python

    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
"""


# ── LLM response type ──────────────────────────────────────────────────


class ToolCall:
    """A single tool-call request from the LLM."""

    def __init__(
        self,
        id: str,
        name: str,
        arguments: Dict[str, Any],
    ) -> None:
        self.id = id
        self.name = name
        self.arguments = arguments

    def __repr__(self) -> str:
        return f"ToolCall(id={self.id!r}, name={self.name!r})"


LLMResponse = Dict[str, Any]
"""Expected shape of the LLM response:

.. code-block:: python

    {
        "content": "The answer is...",           # Optional text
        "tool_calls": [                           # Optional tool calls
            {
                "id": "call_xxx",
                "name": "search",
                "arguments": {"query": "hello"}
            }
        ]
    }
"""


# ── Tool registry ──────────────────────────────────────────────────────


class ToolRegistry:
    """Registry mapping tool definitions to executable Python functions."""

    def __init__(self, tools: List[ToolDef]) -> None:
        self._tools: Dict[str, ToolDef] = {}
        self._funcs: Dict[str, ToolFunc] = {}
        for t in tools:
            func = t.get("_func")
            fn_info = t.get("function", t)
            name = fn_info.get("name", t.get("name", "unknown"))
            self._tools[name] = t
            if func is not None and callable(func):
                self._funcs[name] = func

    def add_func(self, name: str, func: ToolFunc) -> None:
        self._funcs[name] = func

    def get_definitions(self) -> List[ToolDef]:
        """Return tool definitions with the ``_func`` key stripped (for LLM API)."""
        result = []
        for t in self._tools.values():
            clean = {k: v for k, v in t.items() if k != "_func"}
            result.append(clean)
        return result

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a registered tool and return the result as a string."""
        if name not in self._funcs:
            return f"Error: tool '{name}' not found"
        try:
            result = self._funcs[name](**arguments)
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as e:
            logger.exception("Tool %r failed", name)
            return f"Error executing {name}: {e}"


# ── ToolNode ────────────────────────────────────────────────────────────


def ToolNode(
    llm_func: Callable[[List[Dict[str, Any]], List[ToolDef]], LLMResponse],
    tools: Optional[List[ToolDef]] = None,
    *,
    state_messages_field: str = "messages",
) -> Callable[[Any], Dict[str, Any]]:
    """Create a sync graph node that calls an LLM and executes tool calls.

    The returned callable is a ``NodeFunc`` that can be used directly with
    ``Graph.add_node()``. It implements the standard agent loop:
    call LLM → execute tools → append results.

    Args:
        llm_func:
            A callable ``(messages, tools) -> response`` that takes the current
            message list and tool definitions, and returns an ``LLMResponse``
            dict with optional ``content`` and ``tool_calls`` keys.
        tools:
            A list of tool definitions in OpenAI-compatible format. Each tool
            definition may include a ``_func`` key providing the Python
            callable to execute.
        state_messages_field:
            The state field name that holds the message list (default: ``"messages"``).

    Returns:
        A graph node function ``(state) -> dict``.
    """
    registry = ToolRegistry(tools or [])
    field = state_messages_field

    def _node(state: Any) -> Dict[str, Any]:
        messages = getattr(state, field, [])
        if not isinstance(messages, list):
            messages = []

        # Call LLM
        response = llm_func(messages, registry.get_definitions())
        content = response.get("content", "")
        raw_tool_calls = response.get("tool_calls", [])

        # Build assistant message
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content
        if raw_tool_calls:
            # OpenAI format: {id, function: {name, arguments}} or simplified {id, name, arguments}
            tool_calls_for_msg = []
            for tc in raw_tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name", "") or tc.get("function", {}).get("name", "")
                    tc_args = tc.get("arguments", "") or tc.get("function", {}).get("arguments", "{}")
                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except json.JSONDecodeError:
                            tc_args = {}
                    tool_calls_for_msg.append(ToolCall(id=tc_id, name=tc_name, arguments=tc_args))
                    # For OpenAI-compatible response format
                    if "function" in tc:
                        assistant_msg.setdefault("tool_calls", []).append({
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc_name,
                                "arguments": json.dumps(tc_args, ensure_ascii=False),
                            },
                        })
                    else:
                        assistant_msg["tool_calls"] = raw_tool_calls
                elif isinstance(tc, ToolCall):
                    tool_calls_for_msg.append(tc)

            # Execute tools and collect results
            tool_messages = []
            for tc in tool_calls_for_msg:
                tool_result = registry.execute(tc.name, tc.arguments)
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                    "name": tc.name,
                })
                logger.debug("Tool %r -> %s", tc.name, tool_result[:80] if len(tool_result) > 80 else tool_result)

            return {
                field: Append([assistant_msg] + tool_messages),
            }

        # No tool calls — just append the assistant response
        return {field: Append([assistant_msg])}

    return _node


# ── Router helper ──────────────────────────────────────────────────────


def has_tool_calls(state: Any, field: str = "messages") -> str:
    """Router function for conditional edges.

    Returns ``"tools"`` if the last message contains tool calls,
    ``"end"`` otherwise. Use with :meth:`~graphforge.Graph.add_conditional_edges`.

    Args:
        state: The current graph state.
        field: The state field name holding messages (default: ``"messages"``).

    Returns:
        ``"tools"`` if there are tool calls, ``"end"`` otherwise.
    """
    messages = getattr(state, field, [])
    if not messages:
        return "end"
    last = messages[-1] if isinstance(messages, (list, tuple)) else messages
    if isinstance(last, dict) and (
        "tool_calls" in last or "function_call" in last
    ):
        return "tools"
    return "end"
