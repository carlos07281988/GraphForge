"""
Example 10: Store / Long-Term Memory
Cross-session persistent memory for agents.
"""
from graphforge import Graph, GraphState, node_field, Append, configure_logging
from graphforge.store import InMemoryStore

configure_logging()


class MemoryState(GraphState):
    user_id: str = ""
    query: str = ""
    response: str = ""
    history: list = node_field(default=[], merge="append")


def process_with_memory(state: MemoryState, store):
    """Node that reads/writes user preferences from the store."""
    user_id = state.user_id or "default_user"

    # Read from store
    prefs = store.get(user_id, "preferences") or {}
    history = store.get(user_id, "history") or []

    # Generate response incorporating stored knowledge
    name = prefs.get("name", "User")
    lang = prefs.get("language", "en")
    greeting = f"Hello {name}!" if lang == "en" else f"你好 {name}!"

    response = f"{greeting} You asked: {state.query}"

    # Update store with new information
    history.append({"query": state.query, "response": response})
    store.put(user_id, "history", history)

    if "my name is" in state.query.lower():
        name = state.query.lower().split("my name is")[-1].strip().capitalize()
        prefs["name"] = name
        store.put(user_id, "preferences", prefs)

    return {
        "response": response,
        "history": Append([f"Q: {state.query} / A: {response}"]),
    }


# Build graph
graph = Graph[MemoryState]()
graph.add_node("process", process_with_memory)
graph.add_edge("process", "__end__")
graph.set_entry_point("process")
compiled = graph.compile(state_type=MemoryState)

store = InMemoryStore()

# Simulate multiple conversations
conversations = [
    {"user_id": "alice", "query": "Hi there!"},
    {"user_id": "alice", "query": "my name is Alice"},
    {"user_id": "alice", "query": "What's the weather?"},
    {"user_id": "bob", "query": "Hello!"},
]

for conv in conversations:
    result = compiled.invoke(
        MemoryState(user_id=conv["user_id"], query=conv["query"]),
        store=store,
    )
    print(f"[{conv['user_id']}] {conv['query']}")
    print(f"  → {result.response}")

# Show stored memory
print("\n=== Stored Memory ===")
for user_id in ["alice", "bob"]:
    prefs = store.get(user_id, "preferences") or {}
    history = store.get(user_id, "history") or []
    print(f"\nUser: {user_id}")
    print(f"  Preferences: {prefs}")
    print(f"  History entries: {len(history)}")
    for h in history[-2:]:
        print(f"    - {h}")
