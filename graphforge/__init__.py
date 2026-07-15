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

"""GraphForge — a type-safe, composable graph execution framework.

GraphForge is inspired by LangGraph's state-graph model and guided by
LangChain's design patterns, while deliberately avoiding the pitfalls
of both.
"""

from graphforge._types import (
    AsyncNodeFunc,
    AsyncRouterFunc,
    AsyncStreamingNodeFunc,
    GraphState as GraphStateProtocol,
    NodeFunc,
    NodeName,
    RouterFunc,
    StateUpdate,
    StreamingNodeFunc,
)
from graphforge._graph import CompiledGraph, Graph
from graphforge._node import Node, NodeKind
from graphforge._stream import EventType, StreamEvent, StreamMode
from graphforge._checkpoint import (
    Checkpoint,
    Checkpointer,
    CheckpointKey,
    InMemoryCheckpointer,
)
from graphforge._checkpoint_sqlite import SqliteCheckpointer
from graphforge._checkpoint_redis import RedisCheckpointer
from graphforge._checkpoint_postgres import PostgresCheckpointer
from graphforge._command import Command
from graphforge._interrupt import interrupt
from graphforge._mermaid import export_mermaid
from graphforge._http_server import GraphServer
from graphforge._webhook import WebhookCallback
from graphforge._tracing import TracingCallback
from graphforge._callbacks import Callback, CallbackManager
from graphforge._edge import FanOutEdge
from graphforge._executor import GraphExecutionPaused
from graphforge._visualize import export_dot, render_graph
from graphforge._logging import configure_logging, get_logger
from graphforge.pipeline import Pipeline
from graphforge.state import Append, GraphState, MergeStrategy, node_field
from graphforge.structured_output import with_structured_output, StructuredOutputWrapper
from graphforge.tools import Tool, tool
# Guardrails
from graphforge.guardrails import (
    Guardrail,
    GuardrailAction,
    GuardrailError,
    GuardrailResult,
    FieldLengthGuardrail,
    InputGuardian,
    OutputGuardian,
)
from graphforge.eval import (
    EvalCase,
    EvalResults,
    evaluate,
    exact_match,
    contains,
    json_match,
)
from graphforge.guardrails import (

    Guardrail,
    GuardrailAction,
    GuardrailError,
    GuardrailResult,
    FieldLengthGuardrail,
    InputGuardian,
    OutputGuardian,
)
# Store / Memory
from graphforge.store import Store, InMemoryStore
from graphforge.store_redis import RedisStore
# MapReduce
from graphforge._map_reduce import MapReduce


__all__ = [
    # Graph
    "CompiledGraph",
    "Graph",
    # Node
    "Node",
    "NodeKind",
    # State
    "Append",
    "GraphState",
    "MergeStrategy",
    "node_field",
    # Pipeline
    "Pipeline",
    # Streaming
    "EventType",
    "StreamEvent",
    # Checkpoint
    "Checkpoint",
    "Checkpointer",
    "CheckpointKey",
    "InMemoryCheckpointer",
    "RedisCheckpointer",
    "SqliteCheckpointer",
    # Callbacks
    "Callback",
    "CallbackManager",
    # Edge / Command
    "Command",
    "interrupt",
    "export_mermaid",
    "GraphServer",
    "TracingCallback",
    "END_SENTINEL",
    "FanOutEdge",
    # Visualise
    "export_dot",
    "render_graph",
    # Visualization
    # Logging
    "GraphExecutionPaused",
    "configure_logging",
    "get_logger",
    # Protocol aliases
    "AsyncNodeFunc",
    "AsyncRouterFunc",
    "AsyncStreamingNodeFunc",
    "GraphStateProtocol",
    "NodeFunc",
    "NodeName",
    "RouterFunc",
    "StateUpdate",
    "StreamingNodeFunc",
    # Visualization
    "export_dot",
    "render_graph",
    "Append",
    "Guardrail",
    "GuardrailAction",
    "GuardrailError",
    "GuardrailResult",
    "FieldLengthGuardrail",
    "InputGuardian",
    "OutputGuardian",
    "InMemoryStore",
    "MapReduce",
    "RedisStore",
    "Store",

]

# Version
import logging
_logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)
__author__ = "GraphForge Contributors"
__license__ = "Apache 2.0"
__description__ = __doc__.splitlines()[0].lstrip()

__all__.extend([
    "StreamMode",
    "with_structured_output",
    "StructuredOutputWrapper",
    "Tool",
    "tool",
    "EvalCase",
    "EvalResults",
    "evaluate",
    "exact_match",
    "contains",
    "json_match",
    "PostgresCheckpointer",
])
__all__.sort()
__all__.extend(["__version__", "__version_info__",
    "GraphExecutionPaused", "SqliteCheckpointer"])

_logger.debug("GraphForge %s loaded", __version__)

# Optionally register the a2a subpackage
try:
    from graphforge import a2a  # type: ignore[import-untyped]
except ImportError:
    pass
