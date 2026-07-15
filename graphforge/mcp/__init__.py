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

"""MCP (Model Context Protocol) integration for GraphForge.

This module enables GraphForge agents to connect to any MCP-compatible
server for tool discovery and invocation, and to expose compiled graphs
as MCP endpoints.

Requires ``mcp`` package (install with ``pip install graphforge[mcp]``).
"""

from graphforge.mcp._client import MCPClient
from graphforge.mcp._tool_node import mcp_tools_to_tool_defs, wrap_mcp_tools
from graphforge.mcp._server import MCPAgentServer

__all__ = [
    "MCPClient",
    "MCPAgentServer",
    "mcp_tools_to_tool_defs",
    "wrap_mcp_tools",
]
