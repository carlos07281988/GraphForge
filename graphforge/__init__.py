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
from graphforge._stream import EventType, StreamEvent
from graphforge._checkpoint import (
    Checkpoint,
    Checkpointer,
    CheckpointKey,
    InMemoryCheckpointer,
)
from graphforge._callbacks import Callback, CallbackManager
from graphforge._logging import configure_logging, get_logger
from graphforge.pipeline import Pipeline
from graphforge.state import Append, GraphState, MergeStrategy, node_field

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
    # Callbacks
    "Callback",
    "CallbackManager",
    # Logging
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
]

# Version
import logging
_logger = logging.getLogger(__name__)

__version__ = "0.1.0"
__version_info__ = (0, 1, 0)
__author__ = "GraphForge Contributors"
__license__ = "Apache 2.0"
__description__ = __doc__.splitlines()[0].lstrip()
__all__.sort()
__all__.extend(["__version__", "__version_info__"])

_logger.debug("GraphForge %s loaded", __version__)
