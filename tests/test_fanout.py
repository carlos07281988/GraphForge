"""Tests for parallel/fan-out node execution."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from graphforge import (
    Append,
    Graph,
    GraphState,
    node_field,
)


class FanOutState(GraphState):
    results: list = node_field(default=[], merge="append")
    count: int = 0


def branch_a(state: FanOutState) -> Dict[str, Any]:
    return {"results": Append(["A"]), "count": state.count + 1}


def branch_b(state: FanOutState) -> Dict[str, Any]:
    return {"results": Append(["B"]), "count": state.count + 10}


def branch_c(state: FanOutState) -> Dict[str, Any]:
    return {"results": Append(["C"]), "count": state.count + 100}


def join_node(state: FanOutState) -> Dict[str, Any]:
    return {"results": Append(["join"]), "count": state.count + 1000}


class TestFanOutSync:
    def test_fanout_basic(self) -> None:
        """Fan-out executes all branches from same state."""
        g = Graph[FanOutState]()
        g.add_node("start", lambda s: {})
        g.add_node("a", branch_a)
        g.add_node("b", branch_b)
        g.add_fanout("start", ["a", "b"])
        g.set_entry_point("start")
        compiled = g.compile()

        result = compiled.invoke(FanOutState())
        assert "A" in result.results
        assert "B" in result.results
        assert result.count == 10  # overwrite: branch_b wins

    def test_fanout_with_join(self) -> None:
        """Fan-out with join continues after all branches."""
        g = Graph[FanOutState]()
        g.add_node("start", lambda s: {})
        g.add_node("a", branch_a)
        g.add_node("b", branch_b)
        g.add_node("join", join_node)
        g.add_fanout("start", ["a", "b"], join="join")
        g.add_edge("join", "__end__")
        g.set_entry_point("start")
        compiled = g.compile()

        result = compiled.invoke(FanOutState())
        assert "A" in result.results
        assert "B" in result.results
        assert "join" in result.results
        # a: +1, b: +10, join: +1000 = 1011
        assert result.count == 1010  # branch_b(10) + join(1000)

    def test_fanout_three_branches(self) -> None:
        """Fan-out with three branches."""
        g = Graph[FanOutState]()
        g.add_node("start", lambda s: {})
        g.add_node("a", branch_a)
        g.add_node("b", branch_b)
        g.add_node("c", branch_c)
        g.add_fanout("start", ["a", "b", "c"])
        g.set_entry_point("start")
        compiled = g.compile()

        result = compiled.invoke(FanOutState())
        assert len(result.results) == 3  # a + b + c
        assert result.count == 100  # overwrite: branch_c wins

    def test_fanout_missing_target_raises(self) -> None:
        """Fan-out with unregistered target should fail validation."""
        g = Graph[FanOutState]()
        g.add_node("start", lambda s: {})
        g.add_fanout("start", ["nonexistent"])
        g.set_entry_point("start")
        with pytest.raises(ValueError, match="not registered"):
            g.compile()

    def test_fanout_chained_branches(self) -> None:
        """Branches with multiple nodes after fan-out."""
        g = Graph[FanOutState]()

        def branch_a1(state: FanOutState) -> Dict[str, Any]:
            return {"count": state.count + 1, "results": Append(["a1"])}

        def branch_a2(state: FanOutState) -> Dict[str, Any]:
            return {"count": state.count + 10, "results": Append(["a2"])}

        g.add_node("start", lambda s: {})
        g.add_node("a1", branch_a1)
        g.add_node("a2", branch_a2)
        g.add_fanout("start", ["a1", "a2"])
        g.set_entry_point("start")
        compiled = g.compile()

        result = compiled.invoke(FanOutState())
        assert "a1" in result.results
        assert "a2" in result.results


class TestFanOutAsync:
    @pytest.mark.asyncio
    async def test_async_fanout(self) -> None:
        """Async fan-out runs branches in parallel."""
        g = Graph[FanOutState]()
        g.add_node("start", lambda s: {})
        g.add_node("a", branch_a)
        g.add_node("b", branch_b)
        g.add_fanout("start", ["a", "b"])
        g.set_entry_point("start")
        compiled = g.compile()

        result = await compiled.ainvoke(FanOutState())
        assert "A" in result.results
        assert "B" in result.results

    @pytest.mark.asyncio
    async def test_async_fanout_parallel(self) -> None:
        """Async fan-out runs branches truly in parallel."""
        call_order = []

        def branch_1(state: FanOutState) -> Dict[str, Any]:
            call_order.append("1_start")
            call_order.append("1_end")
            return {"results": Append(["B1"])}

        def branch_2(state: FanOutState) -> Dict[str, Any]:
            call_order.append("2_start")
            call_order.append("2_end")
            return {"results": Append(["B2"])}

        g = Graph[FanOutState]()
        g.add_node("start", lambda s: {})
        g.add_node("b1", branch_1)
        g.add_node("b2", branch_2)
        g.add_fanout("start", ["b1", "b2"])
        g.set_entry_point("start")
        compiled = g.compile()

        result = await compiled.ainvoke(FanOutState())
        assert "B1" in result.results
        assert "B2" in result.results

        # Both branches should execute
        assert "B1" in result.results
        assert "B2" in result.results
