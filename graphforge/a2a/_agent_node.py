# Copyright 2024 GraphForge Contributors
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

"""A2A agent node — outbound agent-to-agent calls from within a graph.

Provides factory functions that create ``NodeFunc`` or ``StreamingNodeFunc``
implementations which call external A2A agents. Use them inside any
``Graph.add_node()`` as you would a regular node.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Optional,
    TypeVar,
    Union,
)

from graphforge import Append, GraphState
from graphforge.a2a._client import A2AClient, SyncA2AClient
from graphforge.a2a._models import (
    A2ATaskError,
    Message,
    Task,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)


_StateT = TypeVar("_StateT", bound=GraphState)
_logger = logging.getLogger(__name__)


# ── Default Mappers ─────────────────────────────────────────────────────────


def _default_input_mapper(state: GraphState) -> Message:
    """Default input mapper: extract messages from state and create A2A message.

    Looks for a ``messages`` attribute on the state. If it's a list of dicts
    with a ``content`` key, the last user message is used.
    """
    msgs = getattr(state, "messages", None)
    if msgs and isinstance(msgs, (list, tuple)):
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                return Message(role="user", parts=[TextPart(text=str(content))])
        # Fallback: use the last message regardless of role
        last = msgs[-1]
        if isinstance(last, dict):
            content = last.get("content", str(last))
        else:
            content = str(last)
        return Message(role="user", parts=[TextPart(text=str(content))])
    return Message(role="user", parts=[TextPart(text=str(state))])


def _default_output_mapper(
    message: Optional[Message],
    task: Task,
) -> Dict[str, Any]:
    """Default output mapper: extract text from A2A response message.

    Returns an ``Append(["response"])`` update for the ``messages`` field.
    """
    text = ""
    if message is not None:
        for part in message.parts:
            if isinstance(part, TextPart):
                text += part.text + "\n"
        text = text.strip()
    if not text:
        text = f"[A2A task {task.id} completed: {task.status.value}]"
    return {"messages": Append([{"role": "assistant", "content": text}])}


# ── Factory: Sync Agent Node ────────────────────────────────────────────────


def create_a2a_agent_node(
    agent_url: str,
    *,
    api_key: Optional[str] = None,
    input_mapper: Optional[
        Callable[[Any], Message]
    ] = None,
    output_mapper: Optional[
        Callable[[Optional[Message], Task], Dict[str, Any]]
    ] = None,
    timeout: float = 30.0,
) -> Callable[[Any], Dict[str, Any]]:
    """Create a sync graph node that calls an external A2A agent.

    The returned callable can be used directly with ``Graph.add_node()``.

    Args:
        agent_url: Base URL of the target A2A agent.
        api_key: Optional bearer token for authentication.
        input_mapper: Converts the current graph state to an A2A ``Message``.
            Defaults to ``_default_input_mapper``.
        output_mapper: Converts the A2A response back to state updates.
            Defaults to ``_default_output_mapper``.
        timeout: Request timeout in seconds.

    Returns:
        A node function ``(state) -> dict``.
    """
    _input_mapper = input_mapper or _default_input_mapper
    _output_mapper = output_mapper or _default_output_mapper
    client = SyncA2AClient(agent_url, api_key=api_key, timeout=timeout)

    def _node(state: Any) -> Dict[str, Any]:
        msg = _input_mapper(state)
        task = client.send_task(msg)
        response_msg = task.messages[-1] if task.messages else None
        return _output_mapper(response_msg, task)

    return _node


# ── Factory: Async Agent Node ───────────────────────────────────────────────


async def _async_node_impl(
    state: Any,
    agent_url: str,
    api_key: Optional[str],
    input_mapper: Callable[[Any], Message],
    output_mapper: Callable[[Optional[Message], Task], Dict[str, Any]],
    timeout: float,
) -> Dict[str, Any]:
    client = A2AClient(agent_url, api_key=api_key, timeout=timeout)
    try:
        msg = input_mapper(state)
        task = await client.send_task(msg)
        response_msg = task.messages[-1] if task.messages else None
        return output_mapper(response_msg, task)
    finally:
        await client.close()


def create_async_a2a_agent_node(
    agent_url: str,
    *,
    api_key: Optional[str] = None,
    input_mapper: Optional[
        Callable[[Any], Message]
    ] = None,
    output_mapper: Optional[
        Callable[[Optional[Message], Task], Dict[str, Any]]
    ] = None,
    timeout: float = 30.0,
) -> Callable[[Any], Any]:
    """Create an async graph node that calls an external A2A agent.

    The returned callable is an ``async def`` function suitable for use
    in async graphs or with ``Graph.add_node()`` (which accepts both sync
    and async functions).

    Args:
        agent_url: Base URL of the target A2A agent.
        api_key: Optional bearer token for authentication.
        input_mapper: Converts the current graph state to an A2A ``Message``.
        output_mapper: Converts the A2A response back to state updates.
        timeout: Request timeout in seconds.

    Returns:
        An async node function ``async (state) -> dict``.
    """
    _input_mapper = input_mapper or _default_input_mapper
    _output_mapper = output_mapper or _default_output_mapper

    async def _async_node(state: Any) -> Dict[str, Any]:
        return await _async_node_impl(
            state, agent_url, api_key,
            _input_mapper, _output_mapper, timeout,
        )

    return _async_node


# ── Factory: Streaming Agent Node ───────────────────────────────────────────


async def _streaming_node_impl(
    state: Any,
    agent_url: str,
    api_key: Optional[str],
    input_mapper: Callable[[Any], Message],
    output_mapper: Callable[[Optional[Message], Task], Dict[str, Any]],
    timeout: float,
) -> AsyncIterator[Dict[str, Any]]:
    client = A2AClient(agent_url, api_key=api_key, timeout=timeout)
    try:
        msg = input_mapper(state)
        final_task: Optional[Task] = None
        async for event in client.send_task_stream(msg):
            _logger.debug(
                "A2A stream event: task=%s status=%s final=%s",
                event.id, event.status, event.final,
            )
            if event.final:
                final_task = Task(
                    id=event.id,
                    status=event.status,
                    messages=[msg, event.message] if event.message else [msg],
                )
                break

        if final_task is None:
            final_task = Task(
                id="unknown",
                status=TaskStatus.COMPLETED,
                messages=[msg],
            )
        response_msg = final_task.messages[-1] if final_task.messages else None
        yield output_mapper(response_msg, final_task)
    finally:
        await client.close()


def create_streaming_a2a_agent_node(
    agent_url: str,
    *,
    api_key: Optional[str] = None,
    input_mapper: Optional[
        Callable[[Any], Message]
    ] = None,
    output_mapper: Optional[
        Callable[[Optional[Message], Task], Dict[str, Any]]
    ] = None,
    timeout: float = 30.0,
) -> Callable[[Any], Any]:
    """Create a streaming async generator node that calls an external A2A agent.

    Uses the A2A ``sendStream`` endpoint to stream task updates. The returned
    callable is an async generator that yields state updates.

    Args:
        agent_url: Base URL of the target A2A agent.
        api_key: Optional bearer token for authentication.
        input_mapper: Converts state to an A2A ``Message``.
        output_mapper: Converts the A2A response to state updates.
        timeout: Request timeout in seconds.

    Returns:
        An async generator node function ``async (state) -> AsyncIterator[dict]``.
    """
    _input_mapper = input_mapper or _default_input_mapper
    _output_mapper = output_mapper or _default_output_mapper

    async def _streaming_node(
        state: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        async for update in _streaming_node_impl(
            state, agent_url, api_key,
            _input_mapper, _output_mapper, timeout,
        ):
            yield update

    return _streaming_node
