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

"""Built-in HTTP server for deploying CompiledGraph as a REST API."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from graphforge._graph import CompiledGraph

try:
    from aiohttp import web
except ImportError:
    web = None

_logger = logging.getLogger(__name__)


class GraphServer:
    """Simple HTTP server that exposes a :class:`~graphforge._graph.CompiledGraph`
    as a REST API.

    Args:
        graph: The compiled graph to serve.
        host: Host to bind (default ``"0.0.0.0"``).
        port: Port to bind (default ``8080``).
        api_key: If set, requires ``Authorization: Bearer <api_key>``.
    """

    def __init__(
        self,
        graph: CompiledGraph[Any],
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        api_key: Optional[str] = None,
    ) -> None:
        if web is None:
            raise ImportError(
                "The ``aiohttp`` package is required. "
                "Install it with: pip install graphforge[a2a]"
            )
        self._graph = graph
        self._host = host
        self._port = port
        self._api_key = api_key
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    # -- build routes -------------------------------------------------------

    def _build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/invoke", self._handle_invoke)
        app.router.add_post("/stream", self._handle_stream)
        app.router.add_get("/health", self._handle_health)
        return app

    # -- lifecycle ----------------------------------------------------------

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

    async def start(self) -> None:
        """Start the server (non-blocking)."""
        self._app = self._build_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        _logger.info("GraphServer started at http://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the server."""
        if self._site is not None:
            await self._site.stop()
        if self._runner is not None:
            await self._runner.cleanup()
        _logger.info("GraphServer stopped")

    def run(self) -> None:
        """Blocking entry point: start the server and run forever."""
        asyncio.run(self._serve())

    async def _serve(self) -> None:
        await self.start()
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()

    async def __aenter__(self) -> GraphServer:
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    # -- routes -------------------------------------------------------------

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "graph": self._graph.name})

    async def _handle_invoke(self, request: web.Request) -> web.Response:
        self._check_auth(request)
        body = await request.json()
        state_data = body.get("state", {})
        config = body.get("config", {})

        # Reconstruct state
        state_type = self._graph.state_type
        if state_type is not None:
            state = state_type.model_validate(state_data)
        else:
            state = state_data

        # Execute
        result = self._graph.invoke(state, config=config)

        # Serialise result
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump(mode="json")
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"result": str(result)}

        return web.json_response(result_dict)

    async def _handle_stream(self, request: web.Request) -> web.StreamResponse:
        self._check_auth(request)
        body = await request.json()
        state_data = body.get("state", {})
        config = body.get("config", {})

        state_type = self._graph.state_type
        if state_type is not None:
            state = state_type.model_validate(state_data)
        else:
            state = state_data

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        try:
            async for event in self._graph.astream(state, config=config):
                payload = {
                    "type": event.type.value,
                    "node": event.node or "",
                    "data": str(event.data) if event.data else "",
                }
                line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await response.write(line.encode("utf-8"))
        except Exception as exc:
            _logger.exception("Stream error")
            error_line = f"data: {json.dumps({'error': str(exc)})}\n\n"
            try:
                await response.write(error_line.encode("utf-8"))
            except Exception:
                pass

        return response


__all__ = ["GraphServer"]
