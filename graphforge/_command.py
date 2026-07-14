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

"""Command — dynamic node-level routing directive.

A :class:`Command` allows a node to dynamically control the execution flow
by specifying both state updates and the next node to execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Command:
    """A directive returned by a node to control the next execution step.

    When a node returns a :class:`Command` instead of a plain ``dict``, the
    executor applies the state *update* and routes to *goto* directly,
    bypassing normal edge resolution.

    Args:
        goto:
            The name of the next node to execute.
        update:
            Optional state updates to apply before the next node runs.
    """

    goto: str
    update: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.update is None:
            self.update = {}

    def __repr__(self) -> str:
        return (
            f"Command(goto={self.goto!r}"
            f"{', update=' + str(self.update) if self.update else ''})"
        )


__all__ = ["Command"]
