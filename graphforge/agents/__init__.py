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
