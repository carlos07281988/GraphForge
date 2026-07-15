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

"""MCP client for connecting to MCP-compatible servers.

Provides :class:`MCPClient` which discovers and invokes tools exposed
by any MCP-compatible server (stdio, SSE, or WebSocket transport).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from graphforge._logging import get_logger

logger = get_logger("mcp.client")

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client

    _HAS_MCP = True
except ImportError:
    _HAS_MCP = False
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Simple MCP tool representation
# ---------------------------------------------------------------------------


class MCPTool:
    """A tool discovered from an MCP server.

    Parameters
    ----------
    name:
        Tool name.
    description:
        Human-readable description.
    input_schema:
        JSON Schema for the tool's input parameters.
    """

    __slots__ = ("name", "description", "input_schema")

    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema or {}

    def __repr__(self) -> str:
        return f"MCPTool(name={self.name!r})"


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------


class MCPClient:
    """Connect to an MCP server and discover/call tools.

    Supports both stdio (subprocess) and SSE (HTTP) transports.

    Parameters
    ----------
    command_or_url:
        For stdio: the shell command to launch (e.g. ``"npx"``).
        For SSE: the URL of the SSE endpoint (e.g. ``"http://localhost:8000/mcp"``).
    args:
        Command-line arguments (stdio only).
    transport:
        ``"stdio"`` (default) or ``"sse"``.
    env:
        Optional environment variables for the subprocess (stdio only).

    Examples
    --------
    .. code-block:: python

        # Connect via stdio
        client = MCPClient("npx", args=["-y", "@modelcontextprotocol/server-filesystem"])

        # Connect via SSE
        client = MCPClient("http://localhost:8000/mcp", transport="sse")

        # List tools
        tools = await client.list_tools()

        # Call a tool
        result = await client.call_tool("read_file", {"path": "/tmp/test.txt"})

        # Use as context manager
        async with MCPClient("npx", args=["..."]) as client:
            tools = await client.list_tools()
    """

    def __init__(
        self,
        command_or_url: str,
        *,
        args: Optional[List[str]] = None,
        transport: str = "stdio",
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        if not _HAS_MCP:
            raise ImportError(
                "The ``mcp`` package is required. "
                "Install with: pip install graphforge[mcp]"
            )
        self._command_or_url = command_or_url
        self._args = args or []
        self._transport = transport
        self._env = env
        self._session: Optional[ClientSession] = None

    # -- lifecycle ----------------------------------------------------------

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        """Establish a connection to the MCP server."""
        if self._transport == "stdio":
            params = StdioServerParameters(
                command=self._command_or_url,
                args=self._args,
                env=self._env,
            )
            self._streams = await stdio_client(params).__aenter__()
            self._session = await ClientSession(
                self._streams[0], self._streams[1]
            ).__aenter__()
            await self._session.initialize()
            logger.info(
                "MCPClient connected via stdio: %s %s",
                self._command_or_url, self._args,
            )
        elif self._transport == "sse":
            self._streams = await sse_client(self._command_or_url).__aenter__()
            self._session = await ClientSession(
                self._streams[0], self._streams[1]
            ).__aenter__()
            await self._session.initialize()
            logger.info("MCPClient connected via SSE: %s", self._command_or_url)
        else:
            raise ValueError(f"Unsupported transport: {self._transport}")

    async def disconnect(self) -> None:
        """Close the connection to the MCP server."""
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._streams is not None:
            await self._streams[0].close()
            await self._streams[1].close()
            self._streams = None

    # -- tool operations ----------------------------------------------------

    async def list_tools(self) -> List[MCPTool]:
        """Discover available tools from the MCP server.

        Returns
        -------
        A list of :class:`MCPTool` instances.
        """
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() first.")
        result = await self._session.list_tools()
        tools = []
        for t in result.tools:
            tools.append(
                MCPTool(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema if hasattr(t, "inputSchema") else {},
                )
            )
        logger.debug("MCPClient.list_tools: found %d tool(s)", len(tools))
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Invoke a tool on the MCP server.

        Parameters
        ----------
        name:
            Tool name.
        arguments:
            Tool input arguments.

        Returns
        -------
        The tool result as a string.
        """
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() first.")
        result = await self._session.call_tool(name, arguments=arguments or {})
        logger.debug("MCPClient.call_tool(%r): %s", name, result)

        # Extract text content
        text_parts = []
        if hasattr(result, "content"):
            for part in result.content:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
                elif hasattr(part, "data") and part.data:
                    text_parts.append(str(part.data))
        return "\n".join(text_parts) if text_parts else str(result)


__all__ = [
    "MCPClient",
    "MCPTool",
]
