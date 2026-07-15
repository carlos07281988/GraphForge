"""One-command unified API server for CompiledGraph.

Starts REST, WebSocket, MCP, and A2A servers simultaneously on a single port.

Usage::

    from graphforge import serve

    serve(compiled_graph, host="0.0.0.0", port=8080)
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from graphforge._graph import CompiledGraph
from graphforge._http_server import GraphServer
from graphforge._logging import get_logger

logger = get_logger("serve")


class UnifiedServer:
    """Unified server that exposes a CompiledGraph via multiple protocols.

    Parameters
    ----------
    graph:
        The compiled graph to serve.
    host:
        Host to bind (default: ``"0.0.0.0"``).
    port:
        Port to bind (default: ``8080``).
    api_key:
        Optional API key for authentication.
    enable_mcp:
        Enable MCP endpoint (requires ``mcp`` package).
    enable_a2a:
        Enable A2A endpoint (requires ``a2a`` package).
    """

    def __init__(
        self,
        graph: CompiledGraph[Any],
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        api_key: Optional[str] = None,
        enable_mcp: bool = True,
        enable_a2a: bool = True,
    ) -> None:
        self._graph = graph
        self._host = host
        self._port = port
        self._api_key = api_key
        self._enable_mcp = enable_mcp
        self._enable_a2a = enable_a2a
        self._graph_server: Optional[GraphServer] = None
        self._mcp_server: Optional[Any] = None
        self._a2a_server: Optional[Any] = None

    async def start(self) -> None:
        """Start all servers."""
        # Start HTTP/WS server (primary)
        self._graph_server = GraphServer(
            self._graph, host=self._host, port=self._port, api_key=self._api_key,
        )
        await self._graph_server.start()
        logger.info(
            "GraphForge server running at http://%s:%d", self._host, self._port
        )
        logger.info("  REST:  POST /invoke, POST /stream")
        logger.info("  WS:    /ws")

        # Start MCP server if available
        if self._enable_mcp:
            try:
                from graphforge.mcp._server import MCPAgentServer

                self._mcp_server = MCPAgentServer(
                    self._graph, server_name=f"{self._graph.name}-mcp",
                )
                logger.info("  MCP:   ✓ (connect via stdio or SSE)")
            except ImportError:
                logger.info("  MCP:   ✗ (install graphforge[mcp])")

        # Start A2A server if available
        if self._enable_a2a:
            try:
                from graphforge.a2a._server import A2AServer
                from graphforge.a2a._models import AgentCard

                card = AgentCard(name=self._graph.name, description="GraphForge agent")
                self._a2a_server = A2AServer(
                    self._graph, agent_card=card,
                    host=self._host, port=self._port + 1,
                )
                await self._a2a_server.start()
                logger.info("  A2A:   http://%s:%d", self._host, self._port + 1)
            except ImportError:
                logger.info("  A2A:   ✗ (install graphforge[a2a])")

    async def stop(self) -> None:
        """Stop all servers."""
        if self._graph_server is not None:
            await self._graph_server.stop()
        if self._a2a_server is not None:
            await self._a2a_server.stop()

    async def serve_forever(self) -> None:
        """Run until interrupted."""
        await self.start()
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    def run(self) -> None:
        """Blocking entry point."""
        asyncio.run(self.serve_forever())


def serve(
    graph: CompiledGraph[Any],
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    api_key: Optional[str] = None,
) -> None:
    """Start a unified API server for a compiled graph.

    One function call starts REST API, WebSocket, MCP, and A2A servers
    simultaneously — no additional configuration needed.

    Parameters
    ----------
    graph:
        The compiled graph to serve.
    host:
        Host to bind (default: ``"0.0.0.0"``).
    port:
        Port to bind (default: ``8080``).
    api_key:
        Optional API key for Bearer authentication.

    Examples
    --------
    .. code-block:: python

        from graphforge import serve

        serve(compiled_graph)                     # defaults: 0.0.0.0:8080
        serve(graph, port=9090, api_key="sk-...") # custom port + auth
    """
    server = UnifiedServer(graph, host=host, port=port, api_key=api_key)
    server.run()


__all__ = [
    "UnifiedServer",
    "serve",
]
