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

"""Multi-agent orchestration patterns built on GraphForge's graph model.

Provides factory functions that create compiled graphs implementing common
multi-agent coordination patterns:

* **Supervisor/Worker** — a single supervisor agent routes tasks to worker
  agents, reviews results, and decides when to finish.
* **Swarm** — agents pass control among themselves; each agent decides
  which (if any) agent runs next.
* **Delegation** — an agent can spawn sub-agents to handle sub-tasks,
  using subgraph composition.

All patterns compose naturally: a pattern's output is a ``CompiledGraph``
that can be used as a subgraph node inside another graph.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

from graphforge import (
    Command,
    Graph,
    GraphState,
    node_field,
)
from graphforge._types import END_SENTINEL
from graphforge._logging import get_logger
from graphforge._types import StateT
from graphforge.state import Append

logger = get_logger("agents.patterns")


# ===================================================================
# Shared state definitions for patterns
# ===================================================================


class SupervisorState(GraphState):
    """State for the Supervisor/Worker pattern.

    Attributes
    ----------
    task:
        The original task description.
    messages:
        Conversation history including supervisor decisions and worker results.
    current_worker:
        Name of the worker currently being routed to.
    iteration:
        Loop counter to prevent infinite loops.
    done:
        Whether the supervisor has determined the task is complete.
    final_answer:
        The final output of the supervisor after all workers finish.
    """

    task: str = ""
    messages: List[Dict[str, Any]] = node_field(default=[], merge="append")
    current_worker: str = ""
    iteration: int = 0
    done: bool = False
    final_answer: str = ""


class SwarmState(GraphState):
    """State for the Swarm pattern.

    Attributes
    ----------
    messages:
        Conversation history across all agents.
    current_agent:
        Name of the currently active agent.
    active_agent_history:
        Ordered list of agents that have been active (for debugging).
    done:
        Whether execution should terminate.
    final_output:
        The final output after all agents have run.
    """

    messages: List[Dict[str, Any]] = node_field(default=[], merge="append")
    current_agent: str = ""
    active_agent_history: List[str] = node_field(default=[], merge="append")
    done: bool = False
    final_output: str = ""


# ===================================================================
# Pattern factories
# ===================================================================


def create_supervisor_worker(
    supervisor_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    workers: Mapping[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
    *,
    max_iterations: int = 10,
    state_type: Optional[Type[GraphState]] = None,
) -> Graph:
    """Create a Supervisor/Worker pattern graph.

    The supervisor receives the task, decides which worker should handle
    it, routes to that worker, then reviews the worker's output and either
    routes again or finishes.

    Parameters
    ----------
    supervisor_fn:
        Callable ``(state_dict) -> dict``. Expected return keys:
        - ``current_worker``: name of the worker to invoke (or ``"__end__"``)
        - ``messages``: optional ``Append([message])`` for the conversation
        - ``done``: boolean indicating whether to finish
        - ``final_answer``: the final answer when done
    workers:
        Mapping of ``worker_name -> callable(state_dict) -> dict``.
    max_iterations:
        Maximum supervisor loops to prevent infinite routing.
    state_type:
        Custom state class (defaults to :class:`SupervisorState`).

    Returns
    -------
    An un-compiled :class:`~graphforge._graph.Graph` ready for ``.compile()``.
    """
    st = state_type or SupervisorState
    graph: Graph = Graph[st]()
    worker_names = list(workers.keys())

    # Supervisor node
    def _supervisor_node(state: st) -> Dict[str, Any]:
        state_dict = _state_to_dict(state)
        result = supervisor_fn(state_dict)

        # Track iteration count
        iteration = getattr(state, "iteration", 0) + 1
        result["iteration"] = iteration

        if iteration >= max_iterations:
            result["done"] = True
            result["current_worker"] = "__end__"
            if "final_answer" not in result:
                result["final_answer"] = "Max iterations reached."

        if result.get("done", False):
            result["current_worker"] = "__end__"
            if "final_answer" not in result:
                result["final_answer"] = str(result.get("messages", []))

        return result

    graph.add_node("supervisor", _supervisor_node)

    # Worker nodes
    for w_name, w_fn in workers.items():
        def _make_worker(fn: Callable) -> Callable:
            def _worker_node(state: st) -> Dict[str, Any]:
                state_dict = _state_to_dict(state)
                return fn(state_dict)
            return _worker_node

        graph.add_node(w_name, _make_worker(w_fn))

    # After supervisor, route to worker or end
    def _route_from_supervisor(state: st) -> str:
        target = getattr(state, "current_worker", "__end__")
        if target == "__end__" or target is None:
            return "__end__"
        if target in worker_names:
            return target
        # Fallback: route to first worker if invalid
        return worker_names[0] if worker_names else "__end__"

    path_map: Dict[str, str] = {w: w for w in worker_names}
    path_map["__end__"] = END_SENTINEL
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {**path_map, END_SENTINEL: END_SENTINEL},
    )

    # After any worker, return to supervisor
    for w_name in worker_names:
        graph.add_edge(w_name, "supervisor")

    graph.set_metadata("agent_type", "supervisor_worker")
    graph.set_metadata("max_iterations", max_iterations)
    graph.set_entry_point("supervisor")

    return graph


def create_swarm(
    agents: Mapping[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
    router_fn: Callable[[Dict[str, Any]], Optional[str]],
    *,
    state_type: Optional[Type[GraphState]] = None,
) -> Graph:
    """Create a Swarm pattern graph.

    Agents pass control to each other via a router function. Each agent
    executes and then the router decides which agent runs next (or ``None``
    to terminate).

    Parameters
    ----------
    agents:
        Mapping of ``agent_name -> callable(state_dict) -> dict``.
    router_fn:
        Callable ``(state_dict) -> agent_name or None``. Return ``None``
        or ``"__end__"`` to terminate.
    state_type:
        Custom state class (defaults to :class:`SwarmState`).

    Returns
    -------
    An un-compiled :class:`~graphforge._graph.Graph` ready for ``.compile()``.
    """
    st = state_type or SwarmState
    graph: Graph = Graph[st]()
    agent_names = list(agents.keys())

    # Register all agent nodes
    for a_name, a_fn in agents.items():
        def _make_swarm_agent(fn: Callable) -> Callable:
            def _agent_node(state: st) -> Dict[str, Any]:
                state_dict = _state_to_dict(state)
                result = fn(state_dict)
                # Track active agent
                result["active_agent_history"] = Append([getattr(state, "current_agent", "unknown")])
                return result
            return _agent_node

        graph.add_node(a_name, _make_swarm_agent(a_fn))

    # Router node
    def _swarm_router(state: st) -> Command:
        state_dict = _state_to_dict(state)
        next_agent = router_fn(state_dict)

        if next_agent is None or next_agent == "__end__" or next_agent == END_SENTINEL:
            return Command(goto=END_SENTINEL, update={"done": True})

        if next_agent not in agent_names:
            return Command(goto=END_SENTINEL, update={"done": True, "final_output": f"Unknown agent: {next_agent}"})

        return Command(
            goto=next_agent,
            update={"current_agent": next_agent},
        )

    graph.add_node("router", _swarm_router)

    # Each agent returns to the router
    for a_name in agent_names:
        graph.add_edge(a_name, "router")
        # Each agent also acts as entry via conditional
    graph.add_edge("router", END_SENTINEL)

    graph.set_metadata("agent_type", "swarm")
    graph.set_metadata("agents", agent_names)
    graph.set_entry_point("router")

    return graph


def create_delegation_agent(
    orchestrator_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    sub_agents: Mapping[str, "Graph"],
    *,
    state_type: Optional[Type[GraphState]] = None,
) -> Graph:
    """Create a Delegation pattern graph.

    An orchestrator agent can delegate sub-tasks to specialized sub-agents
    (themselves compiled graphs). Sub-agents run as subgraphs and return
    results to the orchestrator.

    Parameters
    ----------
    orchestrator_fn:
        Callable ``(state_dict) -> dict``. To delegate, return
        ``{"_delegate": agent_name, ...}``.
    sub_agents:
        Mapping of ``agent_name -> compiled Graph`` for sub-task handling.
    state_type:
        Custom state class (defaults to :class:`SupervisorState`).

    Returns
    -------
    An un-compiled :class:`~graphforge._graph.Graph` ready for ``.compile()``.
    """
    st = state_type or SupervisorState
    graph: Graph = Graph[st]()

    # Orchestrator node
    def _orchestrator(state: st) -> Dict[str, Any]:
        state_dict = _state_to_dict(state)
        return orchestrator_fn(state_dict)

    graph.add_node("orchestrator", _orchestrator)

    # Register sub-agents as nodes
    agent_names = list(sub_agents.keys())
    for a_name, a_graph in sub_agents.items():
        compiled = a_graph.compile(state_type=state_type)
        graph.add_node(a_name, compiled)

    # Router after orchestrator
    def _delegation_router(state: st) -> str:
        state_dict = _state_to_dict(state)
        delegate = state_dict.get("_delegate", "__end__")
        if delegate in agent_names:
            return delegate
        return "__end__"

    path_map: Dict[str, str] = {a: a for a in agent_names}
    path_map["__end__"] = END_SENTINEL
    graph.add_conditional_edges(
        "orchestrator",
        _delegation_router,
        {**path_map, END_SENTINEL: END_SENTINEL},
    )

    # After sub-agent, return to orchestrator
    for a_name in agent_names:
        graph.add_edge(a_name, "orchestrator")

    graph.set_metadata("agent_type", "delegation")
    graph.set_entry_point("orchestrator")

    return graph


# ===================================================================
# Helpers
# ===================================================================


def _state_to_dict(state: Any) -> Dict[str, Any]:
    """Convert a state object to a plain dictionary."""
    if hasattr(state, "model_dump"):
        return state.model_dump()
    if isinstance(state, dict):
        return state
    return dict(state)


__all__ = [
    "SupervisorState",
    "SwarmState",
    "create_supervisor_worker",
    "create_swarm",
    "create_delegation_agent",
]
