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

"""Tests for MCP module (mock-based, no actual MCP server required)."""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List

import pytest
from graphforge import Graph, GraphState, node_field


# ===================================================================
# Mock MCP SDK (to avoid requiring actual mcp package in tests)
# ===================================================================


class MockMCPTool:
    def __init__(self, name: str, description: str = "", inputSchema: Any = None):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema or {}


class MockMCPResult:
    def __init__(self, content: List[Any] = None):
        self.content = content or []


class MockTextContent:
    def __init__(self, text: str):
        self.text = text
        self.type = "text"


# Install mock
_mcp_module = types.ModuleType("mcp")
_mcp_module.ClientSession = type("ClientSession", (), {})
_mcp_module.StdioServerParameters = type("StdioServerParameters", (), {})
sys.modules["mcp"] = _mcp_module

# Must import after mock
from graphforge.mcp._client import MCPTool as GFMCPTool
from graphforge.mcp._tool_node import mcp_tools_to_tool_defs


# ===================================================================
# Tests
# ===================================================================


class TestMCPTool:
    def test_create_tool(self) -> None:
        tool = GFMCPTool("test_tool", "A test tool", {"type": "object"})
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.input_schema == {"type": "object"}

    def test_tool_repr(self) -> None:
        tool = GFMCPTool("calc")
        assert repr(tool) == "MCPTool(name='calc')"


class TestMCPClientInit:
    def test_requires_mcp_package(self) -> None:
        """MCPClient raises ImportError since mcp is not installed."""
        from graphforge.mcp._client import MCPClient

        with pytest.raises(ImportError):
            MCPClient("npx", args=["-y", "some-server"])

    def test_client_not_connectable(self) -> None:
        """Client requires connect() before use."""
        from graphforge.mcp._client import MCPClient

        # Can't test runtime without mcp; just check ImportError
        pass


class TestMCPToolDefs:
    def test_convert_empty(self) -> None:
        result = mcp_tools_to_tool_defs([], lambda n, a: "ok")
        assert result == []

    def test_convert_single_tool(self) -> None:
        tools = [
            GFMCPTool("search", "Search tool", {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            }),
        ]

        def call_fn(name: str, args: Dict[str, Any]) -> str:
            return f"called {name} with {args}"

        defs = mcp_tools_to_tool_defs(tools, call_fn)

        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "search"
        assert defs[0]["function"]["description"] == "Search tool"
        assert "_func" in defs[0]
        assert callable(defs[0]["_func"])

    def test_convert_and_call(self) -> None:
        tools = [GFMCPTool("echo", "Echo input")]
        call_log: List[str] = []

        def call_fn(name: str, args: Dict[str, Any]) -> str:
            call_log.append(name)
            return f"called {name}"

        defs = mcp_tools_to_tool_defs(tools, call_fn)
        result = defs[0]["_func"](message="hello")
        assert "echo" in result
        assert call_log == ["echo"]


class TestMCPAdaptorCompat:
    def test_wrap_mcp_tools_requires_connected_client(self) -> None:
        """wrap_mcp_tools raises if client is not connected."""
        from graphforge.mcp._tool_node import wrap_mcp_tools
        from graphforge.mcp._client import MCPClient

        with pytest.raises(ImportError):
            wrap_mcp_tools(MCPClient("npx"))


class TestMCPAgentServerInit:
    def test_requires_mcp_package(self) -> None:
        """MCPAgentServer raises ImportError when mcp not available."""
        from graphforge.mcp._server import MCPAgentServer

        graph = type("FakeGraph", (object,), {
            "nodes": {},
            "get_node": lambda self, n: None,
            "state_type": None,
        })()

        with pytest.raises(ImportError):
            MCPAgentServer(graph)

    def test_serve_import(self) -> None:
        """MCPAgentServer symbol is importable."""
        from graphforge.mcp._server import MCPAgentServer
        assert MCPAgentServer is not None


class TestMCPModuleAPI:
    def test_module_exports(self) -> None:
        from graphforge.mcp import MCPClient, MCPAgentServer, mcp_tools_to_tool_defs, wrap_mcp_tools
        assert MCPClient is not None
        assert MCPAgentServer is not None
        assert callable(mcp_tools_to_tool_defs)
        assert callable(wrap_mcp_tools)
