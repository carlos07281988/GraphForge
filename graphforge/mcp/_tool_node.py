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

"""Adapt MCP tools to GraphForge's ToolDef format.

Provides :func:`mcp_tools_to_tool_defs` to convert :class:`MCPTool` instances
into GraphForge-compatible tool definitions, and :func:`wrap_mcp_tools` to
create a ready-to-use :class:`ToolNode` from an MCP client.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from graphforge._logging import get_logger
from graphforge.mcp._client import MCPClient, MCPTool

logger = get_logger("mcp.tool_node")


def mcp_tools_to_tool_defs(
    tools: List[MCPTool],
    client_call_fn: Callable[[str, Optional[Dict[str, Any]]], str],
) -> List[Dict[str, Any]]:
    """Convert MCP tools to GraphForge ``ToolDef`` format.

    Each tool is converted to the OpenAI-compatible tool definition that
    GraphForge's ``ToolNode`` expects, with a ``_func`` key that delegates
    to the MCP server via the provided call function.

    Parameters
    ----------
    tools:
        MCP tools discovered from an MCP server.
    client_call_fn:
        An async or sync callable ``(name, arguments) -> str`` for invoking
        the tool on the MCP server.

    Returns
    -------
    A list of ``ToolDef`` dicts ready for use with ``ToolNode``.
    """
    result = []
    for t in tools:
        tool_def = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
            "_func": _make_tool_func(t.name, client_call_fn),
        }
        result.append(tool_def)
    return result


def _make_tool_func(
    name: str,
    client_call_fn: Callable[[str, Optional[Dict[str, Any]]], str],
) -> Callable[..., str]:
    """Create a callable that invokes the MCP tool via the client."""

    def _call(**kwargs: Any) -> str:
        return client_call_fn(name, kwargs)

    return _call


def wrap_mcp_tools(
    client: MCPClient,
    *,
    llm_func: Optional[Callable[..., Any]] = None,
    include_tools: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a ready-to-use ToolNode configuration from an MCP client.

    This is a convenience function that:
    1. Connects to the MCP server (must be pre-connected)
    2. Discovers available tools
    3. Filters by ``include_tools`` / ``exclude_tools``
    4. Converts to GraphForge ``ToolDef`` format
    5. Returns a dict with ``tools`` and ``tool_defs`` keys

    Parameters
    ----------
    client:
        A connected :class:`MCPClient` instance.
    llm_func:
        Optional override for the LLM function (default: auto-create).
    include_tools:
        If set, only include these tool names.
    exclude_tools:
        If set, exclude these tool names.

    Returns
    -------
    A dict with ``tools`` (list of ToolDef) and optionally ``tool_defs``.
    """
    import asyncio
    if hasattr(client, "_session") and client._session is None:
        raise RuntimeError(
            "MCP client is not connected. Call client.connect() first."
        )

    # Discover tools
    tools_list = asyncio.get_event_loop().run_until_complete(
        client.list_tools()
    )

    # Filter
    if include_tools:
        tools_list = [t for t in tools_list if t.name in include_tools]
    if exclude_tools:
        tools_list = [t for t in tools_list if t.name not in exclude_tools]

    # Convert
    client_call = lambda name, args: asyncio.get_event_loop().run_until_complete(
        client.call_tool(name, args)
    )
    tool_defs = mcp_tools_to_tool_defs(tools_list, client_call)

    return {"tools": tool_defs, "tool_defs": tool_defs}


__all__ = [
    "mcp_tools_to_tool_defs",
    "wrap_mcp_tools",
]
