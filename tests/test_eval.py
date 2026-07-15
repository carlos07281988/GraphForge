"""Tests for agent evaluation framework."""
from graphforge import Graph, GraphState, node_field
from graphforge.eval import EvalCase, evaluate, exact_match, contains, json_match


class EvalState(GraphState):
    output: str = ""
    items: list = node_field(default=[], merge="append")


def simple_node(state):
    return {"output": f"processed: {state.output}"}


class TestEval:
    def test_exact_match(self) -> None:
        g = Graph[EvalState]()
        g.add_node("a", simple_node)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(state_type=EvalState)

        cases = [
            EvalCase(
                input={"output": "hello"},
                expected={"output": "processed: hello"},
                metrics=[exact_match("output")],
                name="test1",
            ),
        ]
        results = evaluate(compiled, cases, EvalState)
        assert results.passed == 1

    def test_contains_metric(self) -> None:
        results = evaluate.__class__  # noqa
        metric = contains("text", "hello")
        passed, msg = metric({"text": "hello world"}, {"text": "hello"})
        assert passed
        failed, _ = metric({"text": "goodbye"}, {"text": "hello"})
        assert not failed

    def test_metrics_attribute(self) -> None:
        assert callable(exact_match("x"))
        assert callable(contains("x", "y"))
        assert callable(json_match("x"))
