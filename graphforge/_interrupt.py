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

"""Interrupt — pause graph execution for human input.

Provides :func:`interrupt` which nodes can call to pause execution and
wait for human input via :meth:`~graphforge._graph.CompiledGraph.resume`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from graphforge._executor import GraphExecutionPaused

_logger = logging.getLogger(__name__)


def interrupt(
    *,
    message: str = "Execution interrupted",
    value: Any = None,
    timeout: Optional[float] = None,
    on_timeout: str = "reject",
) -> None:
    """Pause graph execution and wait for human input.

    When a node calls this function, the executor saves a checkpoint and
    returns control to the caller.  Use :meth:`~graphforge._graph.CompiledGraph.resume`
    with the same ``thread_id`` to continue execution.

    Args:
        message: Human-readable description of the interruption.
        value: Optional data to pass to the caller (e.g. a question or
            context that helps the human provide input).
        timeout: Optional timeout in seconds. If the execution is not
            resumed within this time, the ``on_timeout`` action is taken.
        on_timeout: Action on timeout (``"reject"``, ``"approve"``, ``"raise"``).
            Default: ``"reject"``.
    """
    meta: Dict[str, Any] = {}
    if value is not None:
        meta["interrupt_value"] = value
    if timeout is not None:
        meta["interrupt_timeout"] = timeout
        meta["interrupt_on_timeout"] = on_timeout
    raise GraphExecutionPaused(message, metadata=meta if meta else None)


__all__ = ["interrupt"]
