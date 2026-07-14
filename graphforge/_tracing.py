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

"""OpenTelemetry tracing for graph execution.

Provides :class:`TracingCallback`, a :class:`~graphforge._callbacks.Callback`
implementation that creates OpenTelemetry spans for graph and node execution.

Requires ``opentelemetry-api`` (install with ``pip install graphforge[tracing]``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from graphforge._callbacks import Callback

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    trace = None  # type: ignore[assignment]
    Span = None  # type: ignore[assignment,misc]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]

_logger = logging.getLogger(__name__)


class TracingCallback(Callback):
    """Create OpenTelemetry spans for graph and node execution.

    Plug this into any graph invocation via the ``callbacks`` parameter:

    .. code-block:: python

        from graphforge import TracingCallback

        graph.invoke(state, callbacks=CallbackManager([TracingCallback()]))

    Args:
        tracer: An OpenTelemetry tracer instance. If ``None``, uses the
            global ``graphforge`` tracer.
        tracer_name: Name for the tracer (default ``"graphforge"``).
    """

    def __init__(
        self,
        tracer: Optional[Any] = None,
        tracer_name: str = "graphforge",
    ) -> None:
        if not _HAS_OTEL:
            raise ImportError(
                "OpenTelemetry is required for TracingCallback. "
                "Install with: pip install graphforge[tracing]"
            )
        self._tracer: Any = tracer or trace.get_tracer(tracer_name)
        self._graph_span: Optional[Any] = None
        self._node_spans: Dict[str, Any] = {}

    # ── Graph lifecycle ────────────────────────────────────────────────

    def on_graph_start(self, graph_name: str, input_state: dict) -> None:
        span = self._tracer.start_span(f"graph.{graph_name}")
        span.set_attribute("graphforge.graph", graph_name)
        self._graph_span = span

    def on_graph_end(self, graph_name: str, final_state: dict) -> None:
        if self._graph_span is not None:
            self._graph_span.end()
            self._graph_span = None

    def on_graph_error(self, graph_name: str, error: Exception) -> None:
        if self._graph_span is not None:
            self._graph_span.record_exception(error)
            try:
                from opentelemetry.trace import Status, StatusCode
                self._graph_span.set_status(Status(StatusCode.ERROR, str(error)))
            except ImportError:
                pass
            self._graph_span.end()
            self._graph_span = None

    # ── Node lifecycle ─────────────────────────────────────────────────

    def on_node_start(self, node: str, state: dict) -> None:
        span = self._tracer.start_span(f"node.{node}")
        span.set_attribute("graphforge.node", node)
        self._node_spans[node] = span

    def on_node_end(self, node: str, state: dict) -> None:
        span = self._node_spans.pop(node, None)
        if span is not None:
            span.end()

    def on_node_error(self, node: str, error: Exception) -> None:
        span = self._node_spans.pop(node, None)
        if span is not None:
            span.record_exception(error)
            try:
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, str(error)))
            except ImportError:
                pass
            span.end()


__all__ = ["TracingCallback"]
