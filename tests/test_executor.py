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

"""Tests for the execution engine."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from graphforge import (
    Append,
    Graph,
    GraphState,
    InMemoryCheckpointer,
    node_field,
)

from graphforge._executor import SyncExecutor
from graphforge._stream import EventType


class CounterState(GraphState):
    count: int = 0
    path: list = node_field(default=[], merge="append")


def increment(state: CounterState) -> Dict[str, Any]:
    return {"count": state.count + 1, "path": Append(["increment"])}


class TestSyncExecutor:
    def test_execute_simple(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()

        executor = SyncExecutor()
        result = executor.execute(compiled, CounterState(count=0))
        assert result.count == 1

    def test_execute_with_config(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()

        executor = SyncExecutor()
        result = executor.execute(
            compiled, CounterState(count=0), config={"thread_id": "test-1"}
        )
        assert result.count == 1

    def test_stream_yields_events(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()

        executor = SyncExecutor()
        events = list(executor.stream(compiled, CounterState(count=0)))

        assert len(events) >= 2
        assert events[0].type == EventType.GRAPH_START
        assert any(e.type == EventType.NODE_START for e in events)
        assert any(e.type == EventType.NODE_END for e in events)
        assert any(e.type == EventType.GRAPH_END for e in events)

    def test_execute_with_checkpoint(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(checkpointer=InMemoryCheckpointer())

        executor = SyncExecutor()
        result = executor.execute(compiled, CounterState(count=0))
        assert result.count == 1

    def test_stream_records_checkpoints(self) -> None:
        checkpointer = InMemoryCheckpointer()
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(checkpointer=checkpointer)

        executor = SyncExecutor()
        list(executor.stream(compiled, CounterState(count=0)))

        keys = checkpointer.list("default")
        assert len(keys) >= 1

    def test_recursion_limit_in_executor(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "a")
        g.set_entry_point("a")
        compiled = g.compile()

        executor = SyncExecutor()
        with pytest.raises(RecursionError, match="recursion limit"):
            executor.execute(compiled, CounterState(count=0), config={"recursion_limit": 3})

    def test_stream_recursion_limit(self) -> None:
        g = Graph[CounterState]()
        g.add_node("a", increment)
        g.add_edge("a", "a")
        g.set_entry_point("a")
        compiled = g.compile()

        executor = SyncExecutor()
        with pytest.raises(RecursionError, match="recursion limit"):
            list(executor.stream(compiled, CounterState(count=0), config={"recursion_limit": 3}))
