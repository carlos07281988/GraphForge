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

"""A2A (Agent-to-Agent) protocol HTTP client.

Provides an async HTTP client that discovers remote agents via their
Agent Card and communicates with them using the A2A task protocol.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from graphforge.a2a._models import (
    A2AAuthenticationError,
    A2AConnectionError,
    A2AProtocolError,
    A2ATaskError,
    AgentCard,
    Message,
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

try:
    import aiohttp
except ImportError:
    aiohttp = None


_logger = logging.getLogger(__name__)


class A2AClient:
    """Async HTTP client for the A2A protocol.

    Discovers a remote agent's capabilities via ``/.well-known/agent-card``
    and provides methods to send tasks, stream results, query status, and
    cancel tasks.

    Args:
        agent_url: Base URL of the remote agent (e.g. ``"http://localhost:8080"``).
        api_key: Optional bearer token for authenticated requests.
        timeout: Default request timeout in seconds.
    """

    def __init__(
        self,
        agent_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        if aiohttp is None:
            raise ImportError(
                "The `aiohttp` package is required for A2A client support. "
                "Install it with: pip install graphforge[a2a]"
            )
        self._agent_url = agent_url.rstrip("/")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._agent_card: Optional[AgentCard] = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers: Dict[str, str] = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=self._timeout,
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "A2AClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ── Agent Discovery ──────────────────────────────────────────────────

    async def fetch_agent_card(self) -> AgentCard:
        """Fetch and cache the remote agent's Agent Card from
        ``/.well-known/agent-card``.
        """
        session = await self._get_session()
        try:
            async with session.get(
                f"{self._agent_url}/.well-known/agent-card"
            ) as resp:
                if resp.status == 401:
                    raise A2AAuthenticationError(
                        f"Authentication required for {self._agent_url}"
                    )
                if resp.status != 200:
                    raise A2AProtocolError(
                        f"Agent card fetch returned {resp.status} "
                        f"from {self._agent_url}"
                    )
                data = await resp.json()
                self._agent_card = AgentCard.model_validate(data)
                _logger.info(
                    "Discovered agent '%s' (version %s) at %s",
                    self._agent_card.name,
                    self._agent_card.version,
                    self._agent_url,
                )
                return self._agent_card
        except aiohttp.ClientError as exc:
            raise A2AConnectionError(
                f"Failed to connect to {self._agent_url}: {exc}"
            ) from exc

    def get_cached_agent_card(self) -> Optional[AgentCard]:
        """Return the cached Agent Card, or ``None`` if not yet fetched."""
        return self._agent_card

    # ── Task Operations ──────────────────────────────────────────────────

    async def send_task(
        self,
        message: Message,
        *,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Send a message to the agent and wait for the final task result.

        Args:
            message: The message to send (role + parts).
            task_id: Optional explicit task ID (auto-generated if omitted).
            metadata: Optional task metadata.

        Returns:
            The completed task with final status.

        Raises:
            A2ATaskError: If the task finishes with ``failed`` or ``canceled``.
        """
        request = TaskSendRequest(
            id=task_id or str(uuid.uuid4()),
            message=message,
            metadata=metadata or {},
        )
        session = await self._get_session()
        try:
            async with session.post(
                f"{self._agent_url}/tasks/send",
                json=request.model_dump(mode="json", by_alias=True),
            ) as resp:
                data = await resp.json()
                if resp.status == 401:
                    raise A2AAuthenticationError(
                        f"Authentication required for {self._agent_url}"
                    )
                if resp.status != 200:
                    raise A2AProtocolError(
                        f"tasks/send returned {resp.status}: {data}"
                    )
                response = TaskSendResponse.model_validate(data)
                task = response.task
                if task.status in (TaskStatus.FAILED, TaskStatus.CANCELED):
                    last_msg = task.messages[-1] if task.messages else None
                    detail = ""
                    if last_msg and last_msg.parts:
                        parts_text = [
                            p.text
                            for p in last_msg.parts
                            if hasattr(p, "text")
                        ]
                        detail = "; ".join(parts_text)
                    raise A2ATaskError(
                        f"Task {task.id} finished with status "
                        f"'{task.status.value}'"
                        + (f": {detail}" if detail else "")
                    )
                return task
        except aiohttp.ClientError as exc:
            raise A2AConnectionError(
                f"Failed to send task to {self._agent_url}: {exc}"
            ) from exc

    async def send_task_stream(
        self,
        message: Message,
        *,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[TaskStatusUpdateEvent]:
        """Send a message and stream task status updates via SSE.

        Yields ``TaskStatusUpdateEvent`` as the agent processes the task.
        The final event has ``final=True``.

        Args:
            message: The message to send.
            task_id: Optional explicit task ID.
            metadata: Optional task metadata.
        """
        request = TaskSendRequest(
            id=task_id or str(uuid.uuid4()),
            message=message,
            metadata=metadata or {},
        )
        session = await self._get_session()
        try:
            async with session.post(
                f"{self._agent_url}/tasks/sendStream",
                json=request.model_dump(mode="json", by_alias=True),
            ) as resp:
                if resp.status == 401:
                    raise A2AAuthenticationError(
                        f"Authentication required for {self._agent_url}"
                    )
                if resp.status != 200:
                    data = await resp.json()
                    raise A2AProtocolError(
                        f"tasks/sendStream returned {resp.status}: {data}"
                    )
                async for line in resp.content:
                    line = line.strip()
                    if not line or line.startswith(b":") or line.startswith(b"id:"):
                        continue
                    if line.startswith(b"data: "):
                        payload = line[len(b"data: "):]
                        try:
                            parsed = json.loads(payload)
                            event = TaskStatusUpdateEvent.model_validate(
                                parsed
                            )
                            yield event
                            if event.final:
                                return
                        except json.JSONDecodeError:
                            _logger.warning(
                                "Skipping malformed SSE data: %s", payload
                            )
        except aiohttp.ClientError as exc:
            raise A2AConnectionError(
                f"Failed to stream task from {self._agent_url}: {exc}"
            ) from exc

    async def get_task(self, task_id: str) -> Task:
        """Get the current status of a task by ID.

        Args:
            task_id: The task ID to query.
        """
        session = await self._get_session()
        try:
            async with session.get(
                f"{self._agent_url}/tasks/{task_id}"
            ) as resp:
                data = await resp.json()
                if resp.status == 404:
                    raise A2AProtocolError(f"Task {task_id} not found")
                if resp.status != 200:
                    raise A2AProtocolError(
                        f"GET tasks/{task_id} returned {resp.status}: {data}"
                    )
                response = TaskGetResponse.model_validate(data)
                return response.task
        except aiohttp.ClientError as exc:
            raise A2AConnectionError(
                f"Failed to get task {task_id} from {self._agent_url}: {exc}"
            ) from exc

    async def cancel_task(
        self,
        task_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Cancel a running task.

        Args:
            task_id: The task ID to cancel.
            metadata: Optional cancellation metadata.
        """
        request = TaskCancelRequest(metadata=metadata or {})
        session = await self._get_session()
        try:
            async with session.post(
                f"{self._agent_url}/tasks/{task_id}/cancel",
                json=request.model_dump(mode="json"),
            ) as resp:
                data = await resp.json()
                if resp.status == 404:
                    raise A2AProtocolError(f"Task {task_id} not found")
                if resp.status != 200:
                    raise A2AProtocolError(
                        f"POST tasks/{task_id}/cancel returned "
                        f"{resp.status}: {data}"
                    )
                response = TaskCancelResponse.model_validate(data)
                return await self.get_task(task_id)
        except aiohttp.ClientError as exc:
            raise A2AConnectionError(
                f"Failed to cancel task {task_id}: {exc}"
            ) from exc


class SyncA2AClient:
    """Synchronous wrapper around :class:`A2AClient`.

    Convenience wrapper that manages an event loop and async client
    internally. Useful in sync graph nodes.

    Args:
        agent_url: Base URL of the remote agent.
        api_key: Optional bearer token.
        timeout: Default request timeout in seconds.
    """

    def __init__(
        self,
        agent_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self._client = A2AClient(agent_url, api_key=api_key, timeout=timeout)

    def fetch_agent_card(self) -> AgentCard:
        """Synchronously fetch the remote agent's Agent Card."""
        return _run_async(self._client.fetch_agent_card())

    def send_task(
        self,
        message: Message,
        *,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Synchronously send a message and wait for the task result."""
        return _run_async(
            self._client.send_task(message, task_id=task_id, metadata=metadata)
        )

    def get_task(self, task_id: str) -> Task:
        """Synchronously get a task's current status."""
        return _run_async(self._client.get_task(task_id))

    def cancel_task(
        self,
        task_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Synchronously cancel a running task."""
        return _run_async(self._client.cancel_task(task_id, metadata=metadata))

    def close(self) -> None:
        """Close the underlying HTTP session."""
        _run_async(self._client.close())


def _run_async(coro):
    """Run a coroutine from synchronous code, reusing the running loop
    if one exists."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(asyncio.ensure_future(coro))
