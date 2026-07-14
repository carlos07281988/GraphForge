"""GraphForge Agents — built-in agent patterns and tool integration.

Provides reusable node types and higher-level agent patterns built on top
of the core graph execution engine.
"""

from graphforge.agents._tool_node import ToolNode, has_tool_calls
from graphforge.agents._react import create_react_agent

__all__ = [
    "ToolNode",
    "has_tool_calls",
    "create_react_agent",
]
