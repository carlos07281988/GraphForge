"""
Example 12: Complex Agent — Full-Featured Graph
Combines streaming, conditional routing, store memory, and evaluation.
"""
from graphforge import (
    Graph, GraphState, node_field, Append,
    InMemoryCheckpointer, Command, CallbackManager,
    configure_logging,
)
from graphforge._callbacks import TimingCallback, CostCallback
from graphforge.store import InMemoryStore
import time

configure_logging()


class AgentState(GraphState):
    user_input: str = ""
    messages: list = node_field(default=[], merge="append")
    context: list = node_field(default=[], merge="append")
    result: str = ""
    confidence: float = 0.0
    needs_help: bool = False
    steps: list = node_field(default=[], merge="append")


# --- Nodes ---

def classify_input(state: AgentState) -> dict:
    text = state.user_input.lower()
    if "help" in text or "?" in text:
        return {"needs_help": True, "steps": Append(["classified as help"])}
    return {"needs_help": False, "steps": Append(["classified as simple"])}


def retrieve_context(state: AgentState) -> dict:
    """Simulate knowledge retrieval."""
    time.sleep(0.05)
    docs = [
        {"content": "GraphForge supports MCP, A2A, RAG, and custom tools.", "score": 0.95},
        {"content": "Nodes can be functions, generators, subgraphs, or pipelines.", "score": 0.87},
    ]
    return {"context": Append(docs), "steps": Append(["retrieved context"])}


def process_simple(state: AgentState) -> dict:
    response = f"Processed: {state.user_input}"
    return {
        "result": response,
        "confidence": 0.9,
        "steps": Append(["simple processing done"]),
    }


def process_with_support(state: AgentState) -> dict:
    context_text = "\n".join(c["content"] for c in state.context) if state.context else "No context"
    response = f"Support response for '{state.user_input}' using:\n{context_text}"
    return {
        "result": response,
        "confidence": 0.95,
        "messages": Append([{"role": "assistant", "content": response}]),
        "steps": Append(["support processing done"]),
    }


def store_result(state: AgentState, store):
    """Store the result in long-term memory."""
    store.put("default", "last_result", {
        "input": state.user_input,
        "result": state.result,
        "confidence": state.confidence,
    })
    history = store.get("default", "history") or []
    history.append({"input": state.user_input, "result": state.result})
    store.put("default", "history", history)
    return {"steps": Append(["stored in memory"])}


def route_after_classify(state: AgentState) -> str:
    if state.needs_help:
        return "support"
    return "simple"


# --- Build Graph ---
def entry(state: AgentState) -> dict:
    """Entry point that branches to classify and retrieve."""
    return {"steps": Append(["started"])}


graph = Graph[AgentState]()
graph.add_node("classify", classify_input)
graph.add_node("retrieve", retrieve_context)
graph.add_node("simple", process_simple)
graph.add_node("support", process_with_support)
graph.add_node("store", store_result)
graph.add_node("entry", entry)

# Conditional routing
graph.add_fanout("entry", ["classify", "retrieve"], join="classify")
graph.add_conditional_edges("classify", route_after_classify, {
    "support": "support",
    "simple": "simple",
})

# Support path also retrieves context
graph.add_edge("support", "store")
graph.add_edge("simple", "store")
graph.add_edge("store", "__end__")

# Retrieve context runs sequentially before classify


graph.set_entry_point("entry")
# Compile
compiled = graph.compile(
    state_type=AgentState,
    checkpointer=InMemoryCheckpointer(),
)

# --- Execute ---
store = InMemoryStore()
timer = TimingCallback()
cm = CallbackManager([timer])

test_inputs = [
    "Hello world",
    "I need help with GraphForge!",
]

for inp in test_inputs:
    result = compiled.invoke(
        AgentState(user_input=inp),
        config={"thread_id": f"session-{hash(inp)}"},
        callbacks=cm,
        store=store,
    )
    print(f"\n=== Input: {inp} ===")
    print(f"  Classified as: {'help' if result.needs_help else 'simple'}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Steps: {' → '.join(result.steps)}")
    print(f"  Result: {result.result[:80]}...")

# Show timing stats
print("\n=== Timing Stats ===")
for node, stats in timer.get_stats().items():
    if node != "_graph_total":
        print(f"  {node}: {stats['duration']:.3f}s ({stats['calls']} calls)")
print(f"  Total: {timer.get_stats().get('_graph_total', {}).get('duration', 0):.3f}s")

# Show stored memory
print("\n=== Stored Memory ===")
history = store.get("default", "history") or []
for h in history:
    print(f"  {h['input']} → {h['result'][:60]}...")
