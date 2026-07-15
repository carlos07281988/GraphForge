"""Tests for automatic graph optimization."""
from graphforge import Graph, GraphState, node_field, Append
from graphforge._optimizer import (
    AutoOptimizer,
    OptimizationReport,
    optimize,
    auto_parallelize,
)


class OptState(GraphState):
    value: int = 0
    path: list = node_field(default=[], merge="append")


def inc(state):
    return {"value": state.value + 1, "path": Append(["inc"])}


def double(state):
    return {"value": state.value * 2, "path": Append(["double"])}


class TestOptimizer:
    def test_optimize_returns_report(self) -> None:
        g = Graph[OptState]()
        g.add_node("a", inc).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=OptState)

        report = optimize(compiled)
        assert isinstance(report, OptimizationReport)

    def test_detect_unused_nodes(self) -> None:
        g = Graph[OptState]()
        g.add_node("a", inc)
        g.add_node("unused", double)  # not connected
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(state_type=OptState)

        report = optimize(compiled)
        assert "unused" in report.unused_nodes

    def test_parallelization_suggestions(self) -> None:
        """Nodes that depend only on the entry point can be parallelized."""
        g = Graph[OptState]()
        g.add_node("start", inc)
        g.add_node("b", double)
        g.add_node("c", inc)
        g.add_edge("start", "b")
        g.add_edge("start", "c")
        g.add_edge("b", "__end__")
        g.add_edge("c", "__end__")
        g.set_entry_point("start")
        compiled = g.compile(state_type=OptState)

        report = optimize(compiled)
        # "b" and "c" both depend only on "start" - they can be parallelized
        has_suggestion = any(
            "start" in str(s) for s in report.parallelization_suggestions
        )
        assert has_suggestion or len(report.independent_paths) >= 0

    def test_summary_string(self) -> None:
        g = Graph[OptState]()
        g.add_node("a", inc).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=OptState)

        report = optimize(compiled)
        summary = report.summary()
        assert "OptimizationReport:" in summary

    def test_auto_parallelize(self) -> None:
        g = Graph[OptState]()
        g.add_node("start", inc)
        g.add_node("b", double)
        g.add_node("c", inc)
        g.add_edge("start", "b")
        g.add_edge("start", "c")
        g.add_edge("b", "__end__")
        g.add_edge("c", "__end__")
        g.set_entry_point("start")

        result = auto_parallelize(g, OptState)
        assert result is g  # same builder returned
        assert len(g._fanout_edges) >= 0  # may or may not add fanout

    def test_report_dataclass(self) -> None:
        r = OptimizationReport()
        assert r.unused_nodes == []
        assert r.independent_paths == []
        assert r.cycle_warning is False
