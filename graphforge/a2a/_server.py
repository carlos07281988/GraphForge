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

"""A2A (Agent-to-Agent) protocol HTTP server.

Exposes a ``CompiledGraph`` as an A2A-compatible HTTP endpoint, enabling
other agents (regardless of framework) to discover and invoke it via the
standard A2A protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

from graphforge import CompiledGraph, EventType, GraphState, StreamEvent
from graphforge.a2a._models import (
    A2AProtocolError,
    AgentCard,
    Message,
    Task,
    TaskCancelResponse,
    TaskGetResponse,
    TaskSendRequest,
    TaskSendResponse,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

try:
    from aiohttp import web
except ImportError:
    web = None


_StateT = TypeVar("_StateT", bound=GraphState)
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(message: Message) -> str:
    texts: List[str] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            texts.append(part.text)
    return "\n".join(texts)


def _default_state_factory(message: Message) -> Dict[str, Any]:
    """Default mapper: creates a state dict with the message text content."""
    return {
        "messages": [{"role": message.role, "content": _extract_text(message)}]
    }


def _default_result_mapper(state: GraphState) -> Message:
    """Default mapper: extracts the last assistant message from state."""
    msgs = getattr(state, "messages", None)
    if msgs and isinstance(msgs, (list, tuple)):
        last = msgs[-1]
        if isinstance(last, dict):
            content = str(last.get("content", last))
        else:
            content = str(last)
        return Message(role="agent", parts=[TextPart(text=content)])
    return Message(role="agent", parts=[TextPart(text=str(state))])


# ---------------------------------------------------------------------------
# Task store
# ---------------------------------------------------------------------------


class _TaskStore:
    """Thread-safe in-memory store for A2A tasks."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()

    async def add(self, task: Task) -> None:
        async with self._lock:
            self._tasks[task.id] = task

    async def get(self, task_id: str) -> Optional[Task]:
        async with self._lock:
            return self._tasks.get(task_id)

    async def update(self, task_id: str, **updates: Any) -> Optional[Task]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            updated = task.model_copy(update=updates)
            self._tasks[task_id] = updated
            return updated


# ---------------------------------------------------------------------------
# A2A Server
# ---------------------------------------------------------------------------


class A2AServer:
    """A2A-compatible HTTP server that wraps a ``CompiledGraph``.

    Exposes the graph as an A2A agent reachable at ``http://<host>:<port>/``.
    Other agents can discover it via ``/.well-known/agent-card`` and invoke
    it via the standard task endpoints.

    Args:
        graph: The compiled graph to expose.
        agent_card: Agent Card describing this agent's identity and capabilities.
        state_factory: Callable that creates a ``GraphState`` (or dict) from
            an incoming A2A ``Message``. Defaults to ``_default_state_factory``.
        result_mapper: Callable that converts the final ``GraphState`` to an
            A2A ``Message``. Defaults to ``_default_result_mapper``.
        host: Host to bind the HTTP server on.
        port: Port to bind the HTTP server on.
        api_key: If set, the server requires ``Authorization: Bearer <api_key>``
            on all endpoints.
    """

    def __init__(
        self,
        graph: CompiledGraph[_StateT],
        *,
        agent_card: AgentCard,
        state_factory: Optional[
            Callable[[Message], Union[_StateT, Dict[str, Any]]]
        ] = None,
        result_mapper: Optional[Callable[[_StateT], Message]] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        api_key: Optional[str] = None,
    ) -> None:
        if web is None:
            raise ImportError(
                "The `aiohttp` package is required for A2A server support. "
                "Install it with: pip install graphforge[a2a]"
            )

        self._graph = graph
        self._agent_card = agent_card
        self._state_factory = state_factory or _default_state_factory
        self._result_mapper = result_mapper or _default_result_mapper
        self._host = host
        self._port = port
        self._api_key = api_key

        self._task_store = _TaskStore()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    # ------------------------------------------------------------------
    # Build routes
    # ------------------------------------------------------------------

    def _build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/.well-known/agent-card", self._handle_agent_card)
        app.router.add_post("/tasks/send", self._handle_task_send)
        app.router.add_post("/tasks/sendStream", self._handle_task_send_stream)
        app.router.add_get("/tasks/{task_id}", self._handle_task_get)
        app.router.add_post("/tasks/{task_id}/cancel", self._handle_task_cancel)
        return app

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the HTTP server (non-blocking). Call ``stop()`` to shut down."""
        if self._app is None:
            self._app = self._build_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        _logger.info(
            "A2A server started at http://%s:%d (agent: %s)",
            self._host,
            self._port,
            self._agent_card.name,
        )

    async def stop(self) -> None:
        """Stop the HTTP server and clean up resources."""
        if self._site is not None:
            await self._site.stop()
        if self._runner is not None:
            await self._runner.cleanup()
        _logger.info("A2A server stopped")

    def run(self) -> None:
        """Blocking entry point: start the server and run forever."""
        asyncio.run(self._run_forever())

    async def _run_forever(self) -> None:
        await self.start()
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    async def __aenter__(self) -> A2AServer:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _check_auth(self, request: web.Request) -> None:
        if self._api_key is None:
            return
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {self._api_key}":
            raise web.HTTPUnauthorized(
                headers={"WWW-Authenticate": "Bearer"},
                text=json.dumps({"error": "Unauthorized"}),
                content_type="application/json",
            )

    # ------------------------------------------------------------------
    # Route: /.well-known/agent-card
    # ------------------------------------------------------------------

    async def _handle_agent_card(self, request: web.Request) -> web.Response:
        self._check_auth(request)
        return web.json_response(
            self._agent_card.model_dump(mode="json", by_alias=True),
            content_type="application/json",
        )

    # ------------------------------------------------------------------
    # Route: POST /tasks/send
    # ------------------------------------------------------------------

    async def _handle_task_send(self, request: web.Request) -> web.Response:
        self._check_auth(request)
        body = await request.json()
        req = TaskSendRequest.model_validate(body)
        task_id = req.id or str(uuid.uuid4())

        task = Task(
            id=task_id,
            status=TaskStatus.WORKING,
            messages=[req.message],
            metadata={**req.metadata, "_push_url": req.push_notification.url}
                   if req.push_notification else req.metadata,
        )
        await self._task_store.add(task)

        try:
            state = self._build_state(req.message)
            result_state = self._graph.invoke(state)
            result_msg = self._result_mapper(result_state)

            now = datetime.now(timezone.utc).isoformat()
            task = await self._task_store.update(
                task_id,
                status=TaskStatus.COMPLETED,
                messages=[req.message, result_msg],
                status_changed_at=now,
            )
        except Exception as exc:
            _logger.exception("Task %s failed", task_id)
            now = datetime.now(timezone.utc).isoformat()
            error_msg = Message(
                role="agent", parts=[TextPart(text=str(exc))]
            )
            task = await self._task_store.update(
                task_id,
                status=TaskStatus.FAILED,
                messages=[req.message, error_msg],
                status_changed_at=now,
            )

        # Send push notification if configured
        await self._send_push(task)
        return web.json_response(
            TaskSendResponse(task=task).model_dump(mode="json", by_alias=True),
            status=200,
            content_type="application/json",
        )

    # ------------------------------------------------------------------
    # Route: POST /tasks/sendStream (SSE)
    # ------------------------------------------------------------------

    async def _handle_task_send_stream(
        self, request: web.Request
    ) -> web.StreamResponse:
        self._check_auth(request)
        body = await request.json()
        req = TaskSendRequest.model_validate(body)
        task_id = req.id or str(uuid.uuid4())

        task = Task(
            id=task_id,
            status=TaskStatus.WORKING,
            messages=[req.message],
            metadata={**req.metadata, "_push_url": req.push_notification.url}
                   if req.push_notification else req.metadata,
        )
        await self._task_store.add(task)

        # SSE response headers
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        async def _sse(event: TaskStatusUpdateEvent) -> None:
            payload = event.model_dump(mode="json", by_alias=True)
            line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await response.write(line.encode("utf-8"))

        # Send initial working event
        await _sse(
            TaskStatusUpdateEvent(
                id=task_id, status=TaskStatus.WORKING, final=False
            )
        )

        try:
            state = self._build_state(req.message)
            last_state: Optional[_StateT] = None

            # Stream graph execution and emit SSE events
            async for ev in self._graph.astream(state):
                meta: Dict[str, Any] = {
                    "node": ev.node or "",
                    "event_type": ev.type.value,
                }

                if ev.type == EventType.STATE_UPDATE:
                    if ev.data and isinstance(ev.data, dict):
                        state_data = ev.data.get("state", {})
                        meta["state_snapshot"] = str(state_data)[:500]
                        if state_data:
                            try:
                                st = self._graph.state_type
                                if st is not None:
                                    last_state = st.model_validate(state_data)
                            except Exception:
                                pass

                elif ev.type == EventType.NODE_END:
                    if ev.data and isinstance(ev.data, dict):
                        try:
                            st = self._graph.state_type
                            if st is not None:
                                last_state = st.model_validate(ev.data)
                        except Exception:
                            pass

                elif ev.type == EventType.NODE_ERROR:
                    if ev.data and isinstance(ev.data, dict):
                        err = ev.data.get("error", "unknown")
                        meta["error"] = str(err)

                await _sse(
                    TaskStatusUpdateEvent(
                        id=task_id,
                        status=TaskStatus.WORKING,
                        final=False,
                        metadata=meta,
                    )
                )

            # Determine final state
            final_state: _StateT = last_state or state
            result_msg = self._result_mapper(final_state)

            now = datetime.now(timezone.utc).isoformat()
            await _sse(
                TaskStatusUpdateEvent(
                    id=task_id,
                    status=TaskStatus.COMPLETED,
                    final=True,
                    message=result_msg,
                )
            )
            await self._task_store.update(
                task_id,
                status=TaskStatus.COMPLETED,
                messages=[req.message, result_msg],
                status_changed_at=now,
            )

        except Exception as exc:
            _logger.exception("Streaming task %s failed", task_id)
            error_msg = Message(
                role="agent", parts=[TextPart(text=str(exc))]
            )
            await _sse(
                TaskStatusUpdateEvent(
                    id=task_id,
                    status=TaskStatus.FAILED,
                    final=True,
                    message=error_msg,
                )
            )
            now = datetime.now(timezone.utc).isoformat()
            await self._task_store.update(
                task_id,
                status=TaskStatus.FAILED,
                messages=[req.message, error_msg],
                status_changed_at=now,
            )

        return response

    # ------------------------------------------------------------------
    # Route: GET /tasks/{task_id}
    # ------------------------------------------------------------------

    async def _handle_task_get(self, request: web.Request) -> web.Response:
        self._check_auth(request)
        task_id = request.match_info["task_id"]
        task = await self._task_store.get(task_id)
        if task is None:
            raise web.HTTPNotFound(
                text=json.dumps({"error": f"Task {task_id} not found"}),
                content_type="application/json",
            )
        return web.json_response(
            TaskGetResponse(task=task).model_dump(mode="json", by_alias=True),
            content_type="application/json",
        )

    # ------------------------------------------------------------------
    # Route: POST /tasks/{task_id}/cancel
    # ------------------------------------------------------------------

    async def _handle_task_cancel(self, request: web.Request) -> web.Response:
        self._check_auth(request)
        task_id = request.match_info["task_id"]
        task = await self._task_store.get(task_id)
        if task is None:
            raise web.HTTPNotFound(
                text=json.dumps({"error": f"Task {task_id} not found"}),
                content_type="application/json",
            )
        if task.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELED,
        ):
            return web.json_response(
                TaskCancelResponse(
                    id=task_id, status=task.status
                ).model_dump(mode="json"),
                content_type="application/json",
            )

        now = datetime.now(timezone.utc).isoformat()
        task = await self._task_store.update(
            task_id, status=TaskStatus.CANCELED, status_changed_at=now,
        )
        await self._send_push(task)
        return web.json_response(
            TaskCancelResponse(
                id=task_id, status=TaskStatus.CANCELED,
            ).model_dump(mode="json"),
            content_type="application/json",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ── Push notification helper ─────────────────────────────────────────

    async def _send_push(self, task: Task) -> None:
        push_url = task.metadata.get("_push_url")
        if not push_url:
            return
        try:
            payload = task.model_dump(mode="json", by_alias=True)
            async with aiohttp.ClientSession() as session:
                await session.post(
                    push_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception as exc:
            _logger.warning("Push notification to %s failed: %s", push_url, exc)

    def _build_state(self, message: Message) -> _StateT:
        """Convert an A2A message to a GraphState instance."""
        state_data = self._state_factory(message)
        if isinstance(state_data, GraphState):
            return state_data
        if isinstance(state_data, dict):
            state_type = self._graph.state_type
            if state_type is not None:
                return state_type.model_validate(state_data)
            raise A2AProtocolError(
                "state_factory returned dict but graph has no state_type"
            )
        raise A2AProtocolError(
            f"state_factory returned unexpected type: {type(state_data)}"
        )
