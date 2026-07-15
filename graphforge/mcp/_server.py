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

"""MCP server that exposes a CompiledGraph as callable MCP tools.

The :class:`MCPAgentServer` wraps a :class:`~graphforge._graph.CompiledGraph`
and exposes each of its nodes as an individual MCP tool, enabling other
agents and MCP clients to invoke graph nodes remotely.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from graphforge._graph import CompiledGraph
from graphforge._logging import get_logger

logger = get_logger("mcp.server")

try:
    from mcp.server import Server as MCPServer
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool as MCPToolSchema, TextContent, CallToolResult

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False
    MCPServer = None  # type: ignore[assignment]
    MCPToolSchema = None  # type: ignore[assignment]
    TextContent = None  # type: ignore[assignment]
    CallToolResult = None  # type: ignore[assignment]


class MCPAgentServer:
    """Expose a :class:`~graphforge._graph.CompiledGraph` as MCP tools.

    Each graph node becomes a callable MCP tool. The server advertises
    available tools via the standard MCP ``tools/list`` endpoint, and
    node invocations happen via ``tools/call``.

    Parameters
    ----------
    graph:
        The compiled graph to expose.
    server_name:
        Name for the MCP server (default: ``"graphforge-agent"``).
    include_nodes:
        If set, only expose these node names as tools.
    exclude_nodes:
        If set, exclude these node names from exposure.

    Examples
    --------
    .. code-block:: python

        from graphforge.mcp import MCPAgentServer

        server = MCPAgentServer(compiled_graph)
        server.serve_stdio()

        # Or for SSE transport:
        # server.serve_sse(host="0.0.0.0", port=8000)
    """

    def __init__(
        self,
        graph: CompiledGraph[Any],
        *,
        server_name: str = "graphforge-agent",
        include_nodes: Optional[List[str]] = None,
        exclude_nodes: Optional[List[str]] = None,
    ) -> None:
        if not _HAS_MCP:
            raise ImportError(
                "The ``mcp`` package is required for MCPAgentServer. "
                "Install with: pip install graphforge[mcp]"
            )
        self._graph = graph
        self._server_name = server_name
        self._include_nodes = include_nodes
        self._exclude_nodes = exclude_nodes

        # Build the list of exposed nodes
        self._node_names = self._get_exposed_nodes()

        # Create MCP server
        self._server = MCPServer(server_name)

        # Register MCP handlers
        self._register_handlers()

    def _get_exposed_nodes(self) -> List[str]:
        """Determine which nodes to expose as MCP tools."""
        names = list(self._graph.nodes.keys())
        if self._include_nodes:
            names = [n for n in names if n in self._include_nodes]
        if self._exclude_nodes:
            names = [n for n in names if n not in self._exclude_nodes]
        # Always exclude __end__ sentinel
        names = [n for n in names if n != "__end__"]
        return names

    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self._server.list_tools()
        async def list_tools() -> List[MCPToolSchema]:
            tools = []
            for name in self._node_names:
                node = self._graph.get_node(name)
                # Extract metadata for description and schema
                metadata = node.metadata or {}
                description = metadata.get("description", f"Graph node: {name}")
                input_schema = metadata.get("input_schema", {
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "object",
                            "description": "Graph state input",
                        }
                    },
                })
                tools.append(
                    MCPToolSchema(
                        name=name,
                        description=description,
                        inputSchema=input_schema,
                    )
                )
            logger.debug(
                "MCPAgentServer.list_tools: %d tools", len(tools)
            )
            return tools

        @self._server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            if name not in self._graph.nodes:
                raise ValueError(f"Unknown node: {name}")

            node = self._graph.get_node(name)
            state_type = self._graph.state_type

            # Reconstruct state from arguments
            state_data = arguments.get("state", arguments)
            if state_type is not None:
                state = state_type.model_validate(state_data)
            else:
                from graphforge.state import GraphState as GS
                state = GS.model_validate(state_data)

            # Invoke the node
            updates = node.invoke(state)

            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(updates, default=str, ensure_ascii=False))]
            )

    # -- serving ------------------------------------------------------------

    async def serve_stdio(self) -> None:
        """Serve over stdio transport (for direct subprocess communication)."""
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream, write_stream, self._server.create_initialization_options()
            )

    def serve(self) -> None:
        """Blocking entry point for stdio transport."""
        import asyncio
        asyncio.run(self.serve_stdio())


__all__ = [
    "MCPAgentServer",
]
