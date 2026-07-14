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

"""Webhook callback — sends HTTP POST on graph lifecycle events.

Provides :class:`WebhookCallback`, a :class:`~graphforge._callbacks.Callback`
implementation that notifies an external endpoint on graph events.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set
from urllib.error import URLError
from urllib.request import Request, urlopen

from graphforge._callbacks import Callback

_logger = logging.getLogger(__name__)

# ── Event constants ───────────────────────────────────────────────────
EVENT_GRAPH_START = "graph_start"
EVENT_GRAPH_END = "graph_end"
EVENT_GRAPH_ERROR = "graph_error"
EVENT_NODE_START = "node_start"
EVENT_NODE_END = "node_end"
EVENT_NODE_ERROR = "node_error"
EVENT_STATE_UPDATE = "state_update"
EVENT_CONDITIONAL = "conditional"

ALL_EVENTS: Set[str] = {
    EVENT_GRAPH_START,
    EVENT_GRAPH_END,
    EVENT_GRAPH_ERROR,
    EVENT_NODE_START,
    EVENT_NODE_END,
    EVENT_NODE_ERROR,
    EVENT_STATE_UPDATE,
    EVENT_CONDITIONAL,
}


class WebhookCallback(Callback):
    """Send HTTP POST notifications on graph lifecycle events.

    Args:
        url: Target URL for webhook POST requests.
        api_key: Optional ``Bearer`` token sent via ``Authorization`` header.
        events: List of event types to subscribe to (default: all events).
        timeout: HTTP request timeout in seconds (default 10).
    """

    def __init__(
        self,
        url: str,
        *,
        api_key: Optional[str] = None,
        events: Optional[List[str]] = None,
        timeout: float = 10.0,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._events: Set[str] = set(events) if events else ALL_EVENTS.copy()
        self._timeout = timeout

    def _post(self, event: str, data: Dict[str, Any]) -> None:
        if event not in self._events:
            return
        payload = json.dumps({"event": event, "data": data}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            req = Request(self._url, data=payload, headers=headers, method="POST")
            urlopen(req, timeout=self._timeout)
        except URLError as exc:
            _logger.warning("Webhook %s failed: %s", event, exc)

    # ── Callback protocol ────────────────────────────────────────────

    def on_graph_start(self, graph_name: str, input_state: dict) -> None:
        self._post(EVENT_GRAPH_START, {"graph": graph_name})

    def on_graph_end(self, graph_name: str, final_state: dict) -> None:
        self._post(EVENT_GRAPH_END, {"graph": graph_name, "state": final_state})

    def on_graph_error(self, graph_name: str, error: Exception) -> None:
        self._post(EVENT_GRAPH_ERROR, {"graph": graph_name, "error": str(error)})

    def on_node_start(self, node: str, state: dict) -> None:
        self._post(EVENT_NODE_START, {"node": node})

    def on_node_end(self, node: str, state: dict) -> None:
        self._post(EVENT_NODE_END, {"node": node, "state": state})

    def on_node_error(self, node: str, error: Exception) -> None:
        self._post(EVENT_NODE_ERROR, {"node": node, "error": str(error)})

    def on_state_update(self, node: str, updates: dict, new_state: dict) -> None:
        self._post(EVENT_STATE_UPDATE, {"node": node, "updates": updates})

    def on_conditional_edge(self, node: str, result: str, target: str) -> None:
        self._post(EVENT_CONDITIONAL, {"node": node, "routed_to": target})


__all__ = [
    "WebhookCallback",
    "ALL_EVENTS",
    "EVENT_GRAPH_START",
    "EVENT_GRAPH_END",
    "EVENT_GRAPH_ERROR",
    "EVENT_NODE_START",
    "EVENT_NODE_END",
    "EVENT_NODE_ERROR",
    "EVENT_STATE_UPDATE",
    "EVENT_CONDITIONAL",
]
