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

"""A2A (Agent-to-Agent) protocol data models.

Implements Google's Agent-to-Agent (A2A) open protocol message types
as Pydantic v2 models, including AgentCard, Task, Message, Part,
and associated request/response types.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Literal


__all__ = [
    "TaskStatus",
    "TextPart",
    "DataPart",
    "File",
    "FilePart",
    "Part",
    "Message",
    "Artifact",
    "Task",
    "AgentSkill",
    "AgentAuthentication",
    "AgentCapabilities",
    "AgentIcon",
    "AgentProvider",
    "AgentCard",
    "PushNotification",
    "TaskSendRequest",
    "TaskSendResponse",
    "TaskStatusUpdateEvent",
    "TaskGetResponse",
    "TaskCancelRequest",
    "TaskCancelResponse",
    "A2AError",
    "A2AConnectionError",
    "A2AProtocolError",
    "A2AAuthenticationError",
    "A2ATaskError",
]


# ── Exceptions ──────────────────────────────────────────────────────────────


class A2AError(Exception):
    """Base exception for A2A protocol errors."""


class A2AConnectionError(A2AError):
    """Connection to the remote agent failed."""


class A2AProtocolError(A2AError):
    """Protocol-level error (unexpected response, malformed data)."""


class A2AAuthenticationError(A2AError):
    """Authentication with the remote agent failed."""


class A2ATaskError(A2AError):
    """The agent task finished with a failure status."""


# ── Enums ───────────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    """Standard A2A task lifecycle states."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


# ── Content Parts ───────────────────────────────────────────────────────────


class TextPart(BaseModel):
    """A plain-text content part."""

    type: Literal["text"] = "text"
    text: str


class DataPart(BaseModel):
    """A structured-data content part."""

    type: Literal["data"] = "data"
    data: Dict[str, Any]


class FilePart(BaseModel):
    """A file-reference content part."""

    type: Literal["file"] = "file"
    file: "FileRef"


class FileRef(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    """Metadata for a file referenced in a content part."""

    name: Optional[str] = None
    mime_type: Optional[str] = Field(None, alias="mimeType")
    url: Optional[str] = None
    bytes: Optional[str] = None


Part = Union[TextPart, DataPart, FilePart]
"""Discriminated union of all content-part types."""


# ── Messages & Tasks ────────────────────────────────────────────────────────


class Message(BaseModel):
    """A message in an A2A task conversation."""

    role: Literal["user", "agent"]
    parts: List[Part]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    """An artifact produced during task execution."""

    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[Part] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_chunk: bool = Field(True, alias="lastChunk")
    index: int = 0


class Task(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    """An A2A task representing a unit of agent work."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.SUBMITTED
    messages: List[Message] = Field(default_factory=list)
    history: List[Message] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status_changed_at: Optional[str] = Field(None, alias="statusChangedAt")


# ── Agent Card ──────────────────────────────────────────────────────────────


class AgentSkill(BaseModel):
    """A single skill or capability exposed by an agent."""

    id: str
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class AgentAuthentication(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    """Authentication requirements for calling an agent."""

    schemes: List[str] = Field(default_factory=lambda: ["bearer"])
    signing_keys: List[Dict[str, str]] = Field(
        default_factory=list, alias="signingKeys"
    )


class AgentCapabilities(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    """Capabilities advertised by an agent."""

    skills: List[AgentSkill] = Field(default_factory=list)
    streaming: bool = True
    push_notifications: bool = Field(False, alias="pushNotifications")


class AgentIcon(BaseModel):
    """Icon resource for an agent card."""

    url: str
    width: Optional[int] = None
    height: Optional[int] = None


class AgentProvider(BaseModel):
    """Provider/organisation behind an agent."""

    organization: Optional[str] = None
    url: Optional[str] = None


class AgentCard(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    """Machine-readable agent discovery document.

    Follows the A2A Agent Card specification for agent discovery
    and capability advertisement.
    """

    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    provider: Optional[AgentProvider] = None
    version: str = "1.0.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    authentication: Optional[AgentAuthentication] = None
    default_input_modes: List[str] = Field(
        ["text"], alias="defaultInputModes"
    )
    default_output_modes: List[str] = Field(
        ["text"], alias="defaultOutputModes"
    )
    icons: List[AgentIcon] = Field(default_factory=list)


# ── Request / Response Types ────────────────────────────────────────────────


class PushNotification(BaseModel):
    """Push notification configuration for a task."""

    url: str
    authentication: Optional[Dict[str, Any]] = None


class TaskSendRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    """Request to send a message to an agent and create a task."""

    id: Optional[str] = None
    message: Message
    metadata: Dict[str, Any] = Field(default_factory=dict)
    push_notification: Optional[PushNotification] = Field(
        None, alias="pushNotification"
    )


class TaskSendResponse(BaseModel):
    """Response from a task send operation."""

    task: Task


class TaskStatusUpdateEvent(BaseModel):
    """SSE event payload for task status updates (sendStream)."""

    id: str
    status: TaskStatus
    final: bool = False
    message: Optional[Message] = None
    artifacts: List[Artifact] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskGetResponse(BaseModel):
    """Response from a task status query."""

    task: Task


class TaskCancelRequest(BaseModel):
    """Request to cancel a running task."""

    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskCancelResponse(BaseModel):
    """Response from a task cancel operation."""

    id: str
    status: TaskStatus
    message: Optional[Message] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
