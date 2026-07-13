"""Tests for the Pipeline module."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from graphforge import GraphState, Pipeline
from graphforge._types import StateUpdate


class DemoState(GraphState):
    value: str = ""
    counter: int = 0


def step_upper(state: DemoState) -> Dict[str, Any]:
    return {"value": state.value.upper()}


def step_counter(state: DemoState) -> Dict[str, Any]:
    return {"counter": state.counter + 1}


class TestPipeline:
    def test_sync_run(self) -> None:
        pipe = Pipeline[DemoState]([step_upper])
        result = pipe.run(DemoState(value="hello"))
        assert result["value"] == "HELLO"

    def test_multi_step(self) -> None:
        pipe = Pipeline[DemoState]([step_upper, step_counter])
        result = pipe.run(DemoState(value="hello", counter=5))
        assert result["value"] == "HELLO"
        assert result["counter"] == 6

    def test_empty_pipeline_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            Pipeline[DemoState]([])

    def test_properties(self) -> None:
        pipe = Pipeline[DemoState]([step_upper], name="test_pipe")
        assert pipe.name == "test_pipe"
        assert len(pipe.steps) == 1
        assert not pipe.is_async()

    def test_repr(self) -> None:
        pipe = Pipeline[DemoState]([step_upper])
        r = repr(pipe)
        assert "Pipeline" in r
        assert "steps=1" in r
