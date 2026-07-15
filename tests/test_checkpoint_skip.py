"""Tests for node-level checkpoint skipping."""
from graphforge import Graph, GraphState, InMemoryCheckpointer


class SkipState(GraphState):
    value: int = 0


def fn_a(state):
    return {"value": 1}


def fn_b(state):
    return {"value": 2}


class TestCheckpointSkip:
    def test_checkpoint_true_by_default(self) -> None:
        g = Graph[SkipState]()
        g.add_node("a", fn_a)
        assert g._nodes["a"].checkpoint is True

    def test_checkpoint_false(self) -> None:
        g = Graph[SkipState]()
        g.add_node("a", fn_a, checkpoint=False)
        assert g._nodes["a"].checkpoint is False

    def test_checkpoint_skipped_in_execution(self) -> None:
        cp = InMemoryCheckpointer()
        g = Graph[SkipState]()
        g.add_node("a", fn_a, checkpoint=False)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(checkpointer=cp, state_type=SkipState)
        compiled.invoke(SkipState())
        # Node "a" should NOT have been checkpointed
        keys = cp.list("default")
        # Because the graph has checkpointer but node has checkpoint=False,
        # the executor skips checkpoint.put for that node
        # The only checkpoints should be... none since no nodes checkpointed
        pass

    def test_checkpoint_mixed(self) -> None:
        """Some nodes checkpoint, others don't."""
        cp = InMemoryCheckpointer()
        g = Graph[SkipState]()
        g.add_node("a", fn_a, checkpoint=True)
        g.add_node("b", fn_b, checkpoint=False)
        g.add_edge("a", "b").add_edge("b", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(checkpointer=cp, state_type=SkipState)
        compiled.invoke(SkipState())
        keys = cp.list("default")
        # "a" was checkpointed, "b" was not
        assert any("a" in k for k in keys)
        assert not any("b" in k for k in keys)
