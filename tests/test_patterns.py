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

"""Tests for multi-agent orchestration patterns."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from graphforge._types import END_SENTINEL
from graphforge import GraphState
from graphforge.agents.patterns import (
    SupervisorState,
    SwarmState,
    create_supervisor_worker,
    create_swarm,
    create_delegation_agent,
)


# ===================================================================
# Supervisor/Worker pattern tests
# ===================================================================


def test_create_supervisor_worker_basic() -> None:
    """Supervisor pattern creates a graph with supervisor and worker nodes."""

    def supervisor(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "current_worker": "worker_a" if state.get("task") == "analyze" else "__end__",
            "done": state.get("task") != "analyze",
            "final_answer": "done" if state.get("task") != "analyze" else "",
        }

    def worker_a(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": [{"role": "assistant", "content": f"analyzed: {state.get('task')}"}]}

    graph = create_supervisor_worker(
        supervisor,
        {"worker_a": worker_a},
        max_iterations=5,
    )
    compiled = graph.compile(state_type=SupervisorState)
    assert "supervisor" in compiled.nodes
    assert "worker_a" in compiled.nodes
    assert compiled.entry_point == "supervisor"
    assert compiled.metadata.get("agent_type") == "supervisor_worker"


def test_supervisor_worker_execution() -> None:
    calls = []

    def supervisor(state: Dict[str, Any]) -> Dict[str, Any]:
        calls.append("supervisor")
        if state.get("iteration", 0) >= 2:
            return {"done": True, "final_answer": "final result", "current_worker": "__end__"}
        return {"current_worker": "worker_a"}

    def worker_a(state: Dict[str, Any]) -> Dict[str, Any]:
        calls.append("worker_a")
        return {"messages": [{"role": "assistant", "content": "processed"}]}

    graph = create_supervisor_worker(supervisor, {"worker_a": worker_a}, max_iterations=10)
    compiled = graph.compile(state_type=SupervisorState)

    result = compiled.invoke(SupervisorState(task="test"))
    assert result.done
    assert "final" in result.final_answer
    # supervisor was called at least 3 times (initial + 2 iterations + final)
    assert calls.count("supervisor") >= 3
    assert calls.count("worker_a") >= 2


def test_supervisor_worker_max_iterations() -> None:
    """When supervisor always routes to a worker, max_iterations should force termination."""

    def supervisor(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"current_worker": "worker_a"}

    def worker_a(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": [{"role": "assistant", "content": "working"}]}

    graph = create_supervisor_worker(supervisor, {"worker_a": worker_a}, max_iterations=3)
    compiled = graph.compile(state_type=SupervisorState)

    result = compiled.invoke(SupervisorState(task="test"))
    assert result.done
    assert result.iteration >= 3


# ===================================================================
# Swarm pattern tests
# ===================================================================


def test_create_swarm_basic() -> None:
    """Swarm pattern creates a graph with agent nodes and a router."""

    def agent_a(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": [{"role": "assistant", "content": "agent_a done"}]}

    def agent_b(state: Dict[str, Any]) -> Dict[str, Any]:
        return {"messages": [{"role": "assistant", "content": "agent_b done"}]}

    def router(state: Dict[str, Any]) -> str:
        if state.get("current_agent") == "agent_a":
            return "agent_b"
        return "__end__"

    graph = create_swarm({"agent_a": agent_a, "agent_b": agent_b}, router)
    compiled = graph.compile(state_type=SwarmState)

    assert "router" in compiled.nodes
    assert "agent_a" in compiled.nodes
    assert "agent_b" in compiled.nodes


def test_swarm_execution() -> None:
    history = []

    def agent_a(state):
        history.append("a")
        return {"messages": [{"role": "assistant", "content": "from a"}]}

    def agent_b(state):
        history.append("b")
        return {"messages": [{"role": "assistant", "content": "from b"}]}

    def router(state):
        if state.get("current_agent") == "agent_a":
            return "agent_b"
        if state.get("current_agent") == "agent_b":
            return None  # terminate
        return "agent_a"

    graph = create_swarm({"agent_a": agent_a, "agent_b": agent_b}, router)
    compiled = graph.compile(state_type=SwarmState)

    result = compiled.invoke(SwarmState())
    assert result.done
    assert len(history) == 2  # agent_a then agent_b


# ===================================================================
# Delegation pattern tests
# ===================================================================


def test_create_delegation_basic() -> None:
    """Delegation pattern creates graph with orchestrator and sub-agents."""
    sub = type("StubGraph", (), {"compile": lambda self, **kw: "compiled"})()
    graph = create_delegation_agent(
        lambda s: {},
        {"sub_a": sub},
    )
    compiled = graph.compile(state_type=SupervisorState)
    assert "orchestrator" in compiled.nodes


def test_patters_import_symbols() -> None:
    """Verify all pattern symbols are accessible."""
    assert SupervisorState is not None
    assert SwarmState is not None
    assert callable(create_supervisor_worker)
    assert callable(create_swarm)
    assert callable(create_delegation_agent)
