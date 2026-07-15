"""Tests for distributed execution."""
from graphforge.distributed import DistributedExecutor
from graphforge import Graph, GraphState


class DState(GraphState):
    value: int = 0


def inc(state):
    return {"value": state.value + 1}


class TestDistributed:
    def test_executor_creates(self) -> None:
        ex = DistributedExecutor(max_workers=2)
        assert ex is not None
        ex.shutdown()

    def test_execute_simple_graph(self) -> None:
        g = Graph[DState]()
        g.add_node("a", inc).add_edge("a", "__end__").set_entry_point("a")
        compiled = g.compile(state_type=DState)
        ex = DistributedExecutor()
        result = ex.execute(compiled, DState())
        assert result.value == 1
        ex.shutdown()

    def test_execute_parallel(self) -> None:
        ex = DistributedExecutor()
        results = ex.execute_parallel([lambda s: {"v": 1}, lambda s: {"v": 2}], [{}, {}])
        assert len(results) == 2
        ex.shutdown()

    def test_execute_node(self) -> None:
        ex = DistributedExecutor()
        future = ex.execute_node(lambda s: {"done": True}, {})
        result = future.result()
        assert result["done"] is True
        ex.shutdown()
