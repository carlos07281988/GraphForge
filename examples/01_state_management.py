"""
Example 01: State Management
Demonstrates different merge strategies: overwrite, append, reduce.
"""
from graphforge import Graph, GraphState, node_field, Append, configure_logging

configure_logging()


# Define state with all merge strategies
class DemoState(GraphState):
    # Overwrite (default): each node replaces the value
    status: str = "idle"

    # Append: new items are added to the list
    log: list = node_field(default=[], merge="append")

    # Reduce: custom function combines old and new values
    counter: int = node_field(default=0, merge="reduce",
                              reducer=lambda old, new: (old or 0) + new)

    # Simple scalar (also overwrite)
    score: float = 0.0


def phase1(state: DemoState) -> dict:
    return {
        "status": "phase1_done",
        "log": Append(["entered phase 1"]),
        "counter": 5,
        "score": 0.5,
    }


def phase2(state: DemoState) -> dict:
    return {
        "status": "phase2_done",
        "log": Append(["entered phase 2"]),
        "counter": 3,
        "score": 0.8,
    }


graph = Graph[DemoState]()
graph.add_node("p1", phase1)
graph.add_node("p2", phase2)
graph.add_edge("p1", "p2")
graph.add_edge("p2", "__end__")
graph.set_entry_point("p1")

compiled = graph.compile()
result = compiled.invoke(DemoState())

print(f"Status:        {result.status}")       # phase2_done (overwrite)
print(f"Log:           {result.log}")           # ['entered phase 1', 'entered phase 2'] (append)
print(f"Counter:       {result.counter}")       # 8 = 5 + 3 (reduce)
print(f"Score:         {result.score}")         # 0.8 (overwrite)

# States are immutable — apply() returns a new snapshot
state1 = DemoState()
state2 = state1.apply(status="running", log=Append(["started"]))
state3 = state2.apply(log=Append(["completed"]), counter=10)
print(f"\nImmutability test:")
print(f"  Original:  {state1.status}, log={state1.log}, counter={state1.counter}")
print(f"  Applied:   {state2.status}, log={state2.log}, counter={state2.counter}")
print(f"  Another:   {state3.status}, log={state3.log}, counter={state3.counter}")
