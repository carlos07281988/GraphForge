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

"""Tests for subgraph checkpoint isolation."""

from __future__ import annotations

from typing import Any, Dict

from graphforge import (
    Graph,
    GraphState,
    InMemoryCheckpointer,
    node_field,
)


class SharedState(GraphState):
    """Shared state for both outer and inner graphs."""
    step: str = ""
    counter: int = 0


def outer_node_a(state: SharedState) -> Dict[str, Any]:
    return {"step": state.step + "a", "counter": state.counter + 1}


def outer_node_b(state: SharedState) -> Dict[str, Any]:
    return {"step": state.step + "b", "counter": state.counter + 10}


def inner_node_x(state: SharedState) -> Dict[str, Any]:
    return {"counter": state.counter + 1, "step": state.step + "x"}


def inner_node_y(state: SharedState) -> Dict[str, Any]:
    return {"counter": state.counter + 2, "step": state.step + "y"}


class TestSubgraphIsolation:
    def test_subgraph_checkpoint_isolation(self) -> None:
        """Subgraph checkpoints use isolated thread_id prefix."""
        # Build inner graph (subgraph)
        inner = Graph[SharedState]()
        inner.add_node("x", inner_node_x)
        inner.add_node("y", inner_node_y)
        inner.add_edge("x", "y")
        inner.add_edge("y", "__end__")
        inner.set_entry_point("x")
        cp = InMemoryCheckpointer()
        inner_compiled = inner.compile(name="inner", checkpointer=cp)
        outer = Graph[SharedState]()
        outer.add_node("a", outer_node_a)
        outer.add_node("sub", inner_compiled)
        outer.add_node("b", outer_node_b)
        outer.add_edge("a", "sub")
        outer.add_edge("sub", "b")
        outer.add_edge("b", "__end__")
        outer.set_entry_point("a")
        compiled = outer.compile(checkpointer=cp)

        result = compiled.invoke(SharedState())
        assert result.counter == 14  # a(+1) → x(+1) → y(+2) → b(+10)

        # Check outer graph checkpoints
        outer_keys = cp.list("default")
        assert len(outer_keys) > 0

        # Check subgraph checkpoints are isolated
        sub_keys = cp.list("sg:sub")
        assert len(sub_keys) > 0
        for key in sub_keys:
            assert key[0] == "sg:sub"  # isolated thread_id

    def test_subgraph_with_own_checkpointer(self) -> None:
        """Subgraph with its own checkpointer uses isolated thread_id."""
        inner_cp = InMemoryCheckpointer()
        inner = Graph[SharedState]()
        inner.add_node("x", inner_node_x)
        inner.add_edge("x", "__end__")
        inner.set_entry_point("x")
        inner_compiled = inner.compile(name="inner", checkpointer=inner_cp)

        outer = Graph[SharedState]()
        outer.add_node("a", outer_node_a)
        outer.add_node("sub", inner_compiled)
        outer.add_edge("a", "sub")
        outer.add_edge("sub", "__end__")
        outer.set_entry_point("a")
        outer_cp = InMemoryCheckpointer()
        compiled = outer.compile(checkpointer=outer_cp)

        result = compiled.invoke(SharedState())
        assert result.counter == 2  # a(+1) → x(+1)

        # Subgraph checkpoints stored on subgraph's checkpointer
        sub_keys = inner_cp.list("sg:sub")
        assert len(sub_keys) > 0

        # Outer checkpoints on outer's checkpointer
        outer_keys = outer_cp.list("default")
        assert len(outer_keys) > 0

    def test_nested_subgraph_isolation(self) -> None:
        """Nested subgraphs each get their own isolated prefix."""
        # Inner-inner subgraph
        innermost = Graph[SharedState]()
        innermost.add_node("x", inner_node_x)
        innermost.add_edge("x", "__end__")
        innermost.set_entry_point("x")
        cp = InMemoryCheckpointer()
        innermost_compiled = innermost.compile(name="innermost", checkpointer=cp)

        # Middle subgraph
        middle = Graph[SharedState]()
        middle.add_node("a", outer_node_a)
        middle.add_node("inner", innermost_compiled)
        middle.add_edge("a", "inner")
        middle.add_edge("inner", "__end__")
        middle.set_entry_point("a")
        middle_compiled = middle.compile(name="middle", checkpointer=cp)

        # Outer graph
        outer = Graph[SharedState]()
        outer.add_node("start", lambda s: {"step": "start", "counter": 1})
        outer.add_node("mid", middle_compiled)
        outer.add_edge("start", "mid")
        outer.add_edge("mid", "__end__")
        outer.set_entry_point("start")

        compiled = outer.compile(checkpointer=cp)

        result = compiled.invoke(SharedState())
        assert result.counter >= 1

        # Middle subgraph checkpoints on parent's cp (shared checkpointer)
        mid_keys = cp.list("sg:mid")
        assert len(mid_keys) > 0
        inner_keys = cp.list("sg:inner")
        assert len(inner_keys) > 0

    def test_subgraph_no_checkpointer(self) -> None:
        """Subgraph without checkpointer uses InMemoryCheckpointer with isolation."""
        inner = Graph[SharedState]()
        inner.add_node("x", inner_node_x)
        inner.add_edge("x", "__end__")
        inner.set_entry_point("x")
        # No checkpointer on inner - it will use InMemoryCheckpointer
        inner_compiled = inner.compile(name="inner")

        outer = Graph[SharedState]()
        outer.add_node("a", outer_node_a)
        outer.add_node("sub", inner_compiled)
        outer.add_edge("a", "sub")
        outer.add_edge("sub", "__end__")
        outer.set_entry_point("a")

        # Only outer has a checkpointer
        cp = InMemoryCheckpointer()
        compiled = outer.compile(checkpointer=cp)

        # Subgraph creates its own InMemoryCheckpointer, which is ephemeral
        # The parent checkpointer doesn't contain subgraph checkpoints
        result = compiled.invoke(SharedState())
        assert result.counter == 2  # a(+1) → x(+1)

        # Only outer checkpoints on parent's checkpointer
        outer_keys = cp.list("default")
        assert len(outer_keys) > 0

        # Subgraph checkpoints should NOT be on parent checkpointer
        # (subgraph uses its own ephemeral InMemoryCheckpointer)
        sub_keys = cp.list("sg:sub")
        assert len(sub_keys) == 0  # Subgraph's InMemoryCheckpointer is local
