"""
Example 05: Multi-Agent Patterns
Demonstrates Supervisor/Worker and Swarm patterns.
"""
from graphforge import GraphState, node_field, Append, configure_logging
from graphforge.agents import create_supervisor_worker, create_swarm, SupervisorState, SwarmState

configure_logging()


# ===== Supervisor/Worker Pattern =====
print("=== Supervisor/Worker Pattern ===")


def supervisor_fn(state_dict: dict) -> dict:
    """Supervisor decides which worker to invoke based on the task."""
    task = state_dict.get("task", "").lower()
    if "search" in task:
        return {"current_worker": "search_worker", "messages": [{"role": "assistant", "content": f"Delegating to search for: {task}"}]}
    elif "analyze" in task:
        return {"current_worker": "analyze_worker", "messages": [{"role": "assistant", "content": f"Delegating to analyze for: {task}"}]}
    else:
        # Done — produce final answer
        return {"done": True, "final_answer": f"Processed task: {task}", "current_worker": "__end__"}


def search_worker_fn(state_dict: dict) -> dict:
    task = state_dict.get("task", "")
    return {"messages": [{"role": "assistant", "content": f"Search results for: {task}"}]}


def analyze_worker_fn(state_dict: dict) -> dict:
    task = state_dict.get("task", "")
    return {"messages": [{"role": "assistant", "content": f"Analysis results for: {task}"}]}


# Build supervisor/worker graph
supervisor_graph = create_supervisor_worker(
    supervisor_fn,
    {"search_worker": search_worker_fn, "analyze_worker": analyze_worker_fn},
    max_iterations=3,
)
supervisor_compiled = supervisor_graph.compile(state_type=SupervisorState)

for task in ["search for AI papers", "analyze customer feedback", "simple task"]:
    result = supervisor_compiled.invoke(SupervisorState(task=task))
    print(f"\nTask: {task}")
    for msg in result.messages:
        if "content" in msg:
            print(f"  [{msg['role']}]: {msg['content']}")
    if result.final_answer:
        print(f"  Final: {result.final_answer}")


# ===== Swarm Pattern =====
print("\n=== Swarm Pattern ===")


def agent_a_fn(state_dict: dict) -> dict:
    return {"messages": [{"role": "assistant", "content": "Agent A: initial processing complete"}]}


def agent_b_fn(state_dict: dict) -> dict:
    return {"messages": [{"role": "assistant", "content": "Agent B: additional analysis complete"}]}


def swarm_router(state_dict: dict) -> str:
    current = state_dict.get("current_agent", "")
    if not current:
        return "agent_a"
    elif current == "agent_a":
        return "agent_b"
    else:
        return "__end__"


swarm_graph = create_swarm({"agent_a": agent_a_fn, "agent_b": agent_b_fn}, swarm_router)
swarm_compiled = swarm_graph.compile(state_type=SwarmState)
result = swarm_compiled.invoke(SwarmState())
print("Swarm execution complete!")
for msg in result.messages:
    print(f"  [{msg['role']}]: {msg['content']}")
print(f"  Active agents: {result.active_agent_history}")
