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

"""Tests for graph visualisation (DOT export and rendering)."""

from __future__ import annotations

import pytest
from graphforge import Graph, GraphState, node_field, export_dot, render_graph
from graphforge._graph import CompiledGraph


class SimpleState(GraphState):
    x: int = 0


class TestExportDot:
    def test_export_simple_graph(self) -> None:
        graph = (
            Graph[SimpleState]()
            .add_node("start", lambda s: {"x": 1})
            .add_edge("start", "__end__")
            .set_entry_point("start")
            .compile()
        )
        dot = export_dot(graph)
        assert "digraph GraphForge" in dot
        assert "start" in dot
        assert "__start__" in dot
        assert "__end__" in dot

    def test_export_shows_label(self) -> None:
        graph = (
            Graph[SimpleState]()
            .add_node("node_a", lambda s: {"x": 1})
            .add_edge("node_a", "__end__")
            .set_entry_point("node_a")
            .compile()
        )
        dot = export_dot(graph, show_kind=False)
        assert '"node_a"' in dot

    def test_export_conditional_edges(self) -> None:
        def router(state: SimpleState) -> str:
            return "odd" if state.x % 2 else "even"

        graph = (
            Graph[SimpleState]()
            .add_node("branch", lambda s: s)
            .add_node("even", lambda s: {"x": s.x // 2})
            .add_node("odd", lambda s: {"x": s.x * 3 + 1})
            .add_conditional_edges(
                "branch",
                router,
                {"even": "even", "odd": "odd"},
            )
            .set_entry_point("branch")
            .compile()
        )
        dot = export_dot(graph)
        assert "branch" in dot
        assert "even" in dot
        assert "odd" in dot

    def test_export_fanout(self) -> None:
        graph = (
            Graph[SimpleState]()
            .add_node("source", lambda s: {"x": 1})
            .add_fanout("source", ["a", "b"], join="join")
            .add_node("a", lambda s: {"x": s.x + 1})
            .add_node("b", lambda s: {"x": s.x + 2})
            .add_node("join", lambda s: {"x": s.x})
            .add_edge("join", "__end__")
            .set_entry_point("source")
            .compile()
        )
        dot = export_dot(graph)
        assert "fan" in dot or "dashed" in dot

    def test_renders_different_graphs(self) -> None:
        g1 = Graph[SimpleState]().add_node("a", lambda s: s).add_edge("a", "__end__").set_entry_point("a").compile()
        g2 = Graph[SimpleState]().add_node("b", lambda s: s).add_edge("b", "__end__").set_entry_point("b").compile()
        assert export_dot(g1) != export_dot(g2)

    def test_export_empty_graph(self) -> None:
        with pytest.raises(ValueError):
            Graph[SimpleState]().compile()


class TestRenderGraph:
    def test_render_without_graphviz_returns_none(self) -> None:
        graph = (
            Graph[SimpleState]()
            .add_node("a", lambda s: s)
            .add_edge("a", "__end__")
            .set_entry_point("a")
            .compile()
        )
        result = render_graph(graph, "/tmp/test_graph.png")
        # May be None if graphviz is not installed
        assert result is None or isinstance(result, str)
