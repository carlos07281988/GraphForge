"""GraphForge A2A — Agent-to-Agent protocol support.

This module provides both **outbound** and **inbound** integration with
Google's Agent-to-Agent (A2A) open protocol:

*Outbound* — call external A2A agents from within a graph using
:func:`create_a2a_agent_node` or :class:`A2AClient`.

*Inbound* — expose a :class:`~graphforge.CompiledGraph` as an A2A-compatible
HTTP endpoint using :class:`A2AServer`.
"""

from graphforge.a2a._models import (
    A2AError,
    A2AConnectionError,
    A2AProtocolError,
    A2AAuthenticationError,
    A2ATaskError,
    AgentAuthentication,
    AgentCapabilities,
    AgentCard,
    AgentIcon,
    AgentProvider,
    AgentSkill,
    Artifact,
    DataPart,
    FilePart,
    FileRef,
    Message,
    Part,
    PushNotification,
    Task,
    TaskCancelRequest,
    TaskCancelResponse,
    TaskGetResponse,
    TaskSendRequest,
    TaskSendResponse,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from graphforge.a2a._client import A2AClient, SyncA2AClient
from graphforge.a2a._server import A2AServer
from graphforge.a2a._agent_node import (
    create_a2a_agent_node,
    create_async_a2a_agent_node,
    create_streaming_a2a_agent_node,
)

__all__ = [
    # Exceptions
    "A2AError",
    "A2AConnectionError",
    "A2AProtocolError",
    "A2AAuthenticationError",
    "A2ATaskError",
    # Models
    "AgentAuthentication",
    "AgentCapabilities",
    "AgentCard",
    "AgentIcon",
    "AgentProvider",
    "AgentSkill",
    "Artifact",
    "DataPart",
    "FilePart",
    "FileRef",
    "Message",
    "Part",
    "PushNotification",
    "Task",
    "TaskCancelRequest",
    "TaskCancelResponse",
    "TaskGetResponse",
    "TaskSendRequest",
    "TaskSendResponse",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "TextPart",
    # Client
    "A2AClient",
    "SyncA2AClient",
    # Server
    "A2AServer",
    # Agent node factories
    "create_a2a_agent_node",
    "create_async_a2a_agent_node",
    "create_streaming_a2a_agent_node",
]
