"""
Example 08: Checkpointing
Persist graph execution state with resumption.
"""
import time
from graphforge import (
    Graph, GraphState, node_field, Append,
    InMemoryCheckpointer, Command, interrupt,
    configure_logging,
)

configure_logging()


class CheckpointState(GraphState):
    messages: list = node_field(default=[], merge="append")
    step: int = 0
    data: str = ""


def step1(state: CheckpointState) -> dict:
    time.sleep(0.1)
    return {
        "messages": Append([{"role": "assistant", "content": "Step 1 complete"}]),
        "step": 1,
        "data": "intermediate_result",
    }


def step2(state: CheckpointState) -> dict:
    time.sleep(0.1)
    return {
        "messages": Append([{"role": "assistant", "content": "Step 2 complete"}]),
        "step": 2,
    }


def pause_for_input(state: CheckpointState) -> dict:
    """Pause and wait for human input (resume later)."""
    print("\n  [Graph paused. Resuming...]")
    # In a real app, the interrupt would pause and wait.
    # On resume, the function continues past the interrupt.
    return {"data": "resumed_with_input", "step": 3,
            "messages": Append([{"role": "assistant", "content": "Resumed after checkpoint"}])}


# Build graph
graph = Graph[CheckpointState]()
graph.add_node("s1", step1)
graph.add_node("s2", step2)
graph.add_node("pause", pause_for_input)
graph.add_edge("s1", "s2")
graph.add_edge("s2", "pause")
graph.add_edge("pause", "__end__")
graph.set_entry_point("s1")

# Compile with checkpointer
checkpointer = InMemoryCheckpointer()
compiled = graph.compile(checkpointer=checkpointer)

# First run
thread_id = "session-1"
print("=== First Execution ===")
result = compiled.invoke(CheckpointState(), config={"thread_id": thread_id})

# Inspect checkpoints
print(f"\nCheckpoints for thread '{thread_id}':")
keys = checkpointer.list(thread_id)
for key in keys:
    cp = checkpointer.get(key)
    print(f"  Step {key[2]}: node={key[1]}, state={cp.state}")

# Resume from the pause
print("\n=== Resuming Execution ===")
result = compiled.resume(thread_id, state_type=CheckpointState,
                         updates={"data": "human_provided_input"})
print(f"\nFinal state:")
for msg in result.messages:
    print(f"  [{msg.get('role', '?')}]: {msg.get('content', '')}")
print(f"  Step: {result.step}")
print(f"  Data: {result.data}")

# Show checkpoint info
print(f"\nCheckpoints for thread '{thread_id}':")
for key in checkpointer.list(thread_id):
    cp = checkpointer.get(key)
    print(f"  Step {key[2]}: node={key[1]}")

print(f"\nCheckpoints for thread 'session-2':")
for key in checkpointer.list("session-2"):
    cp = checkpointer.get(key)
    print(f"  Step {key[2]}: node={key[1]}")
