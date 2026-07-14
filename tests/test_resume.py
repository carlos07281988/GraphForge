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

"""Tests for the resume() API and pause mechanism."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from graphforge import (
    Append,
    Graph,
    GraphExecutionPaused,
    GraphState,
    InMemoryCheckpointer,
    node_field,
)


class ResumeState(GraphState):
    count: int = 0


def increment(state: ResumeState) -> Dict[str, Any]:
    return {"count": state.count + 1}


def pausing_node(state: ResumeState) -> Dict[str, Any]:
    """A node that always pauses."""
    raise GraphExecutionPaused("Awaiting human approval")


class TestResumeAPI:
    def test_resume_continues_from_last_checkpoint(self) -> None:
        """Resume after normal completion returns state as-is."""
        cp = InMemoryCheckpointer()
        g = Graph[ResumeState]()
        g.add_node("a", increment)
        g.add_node("b", increment)
        g.add_edge("a", "b")
        g.add_edge("b", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(checkpointer=cp)

        compiled.invoke(ResumeState(count=0))

        # Resume from last checkpoint (after 'b') - at terminal, no nodes run
        result = compiled.resume("default", state_type=ResumeState)
        assert result.count == 2  # State unchanged from last checkpoint

        # Resume with updates - applied to state before terminal check
        result2 = compiled.resume("default", state_type=ResumeState, updates={"count": 100})
        assert result2.count == 100  # Updates applied, no more nodes to run

    def test_resume_from_pause(self) -> None:
        """Resume re-invokes a paused node."""
        cp = InMemoryCheckpointer()
        g = Graph[ResumeState]()
        g.add_node("a", increment)
        g.add_node("pause", pausing_node)
        g.add_edge("a", "pause")
        g.add_edge("pause", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(checkpointer=cp)

        # Execute - 'a' runs (count=1), 'pause' pauses
        result = compiled.invoke(ResumeState(count=0))
        assert result.count == 1

        # Resume - re-invokes 'pause', which pauses again
        result2 = compiled.resume("default", state_type=ResumeState)
        assert result2.count == 1  # Pause caught, count unchanged

        # Resume with updates - pause re-runs with updated state
        result3 = compiled.resume("default", state_type=ResumeState, updates={"count": 10})
        assert result3.count == 10  # Updates applied, pause caught again

    def test_resume_from_pause_then_continue(self) -> None:
        """After a paused node succeeds on retry, execution continues."""
        cp = InMemoryCheckpointer()

        # A node that pauses once, then succeeds
        class PauseTracker:
            call_count = 0

        def pause_once_then_succeed(state: ResumeState) -> Dict[str, Any]:
            PauseTracker.call_count += 1
            if PauseTracker.call_count == 1:
                raise GraphExecutionPaused("First time pause")
            return {"count": state.count + 10}

        g = Graph[ResumeState]()
        g.add_node("a", increment)
        g.add_node("conditional", pause_once_then_succeed)
        g.add_node("b", increment)
        g.add_edge("a", "conditional")
        g.add_edge("conditional", "b")
        g.add_edge("b", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(checkpointer=cp)

        # First run - pauses at 'conditional'
        result = compiled.invoke(ResumeState(count=0))
        assert result.count == 1  # Only 'a' ran

        # Resume - 'conditional' succeeds this time, then 'b' runs
        result2 = compiled.resume("default", state_type=ResumeState)
        # 'conditional' adds 10, 'b' adds 1 => total 1+10+1 = 12
        assert result2.count == 12

    def test_resume_no_checkpoints_raises(self) -> None:
        """Resuming with no checkpoints raises ValueError."""
        cp = InMemoryCheckpointer()
        g = Graph[ResumeState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(checkpointer=cp)

        with pytest.raises(ValueError, match="No checkpoint"):
            compiled.resume("nonexistent", state_type=ResumeState)

    def test_resume_requires_state_type(self) -> None:
        """Resume must have a state_type."""
        g = Graph[ResumeState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile()

        with pytest.raises(ValueError, match="state_type"):
            compiled.resume("x")

    def test_resume_with_state_type_in_compile(self) -> None:
        """state_type can be set at compile time."""
        cp = InMemoryCheckpointer()
        g = Graph[ResumeState]()
        g.add_node("a", increment)
        g.add_edge("a", "__end__")
        g.set_entry_point("a")
        compiled = g.compile(checkpointer=cp, state_type=ResumeState)

        compiled.invoke(ResumeState(count=0))
        result = compiled.resume("default")
        # After 'a' -> __end__, resume from 'a' sees terminal, returns state
        assert result.count == 1
