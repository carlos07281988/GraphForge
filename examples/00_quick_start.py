"""
Example 00: Quick Start
Minimal GraphForge example — define state, create nodes, build and run a graph.
"""
from graphforge import Graph, GraphState, node_field, Append, configure_logging

configure_logging()


# 1. Define your state
class ChatState(GraphState):
    messages: list = node_field(default=[], merge="append")
    step: str = ""


# 2. Define nodes (functions that receive state and return updates)
def user_proxy(state: ChatState) -> dict:
    """Simulate user input (in a real app, this would be an API endpoint)."""
    return {
        "messages": Append([{"role": "user", "content": "Hello GraphForge!"}]),
        "step": "processed",
    }


def assistant(state: ChatState) -> dict:
    """Process the user message and produce a response."""
    last = state.messages[-1] if state.messages else {}
    content = last.get("content", "")
    return {
        "messages": Append([{"role": "assistant", "content": f"Echo: {content}"}]),
        "step": "done",
    }


# 3. Build the graph
graph = Graph[ChatState]()
graph.add_node("user", user_proxy)
graph.add_node("assistant", assistant)
graph.add_edge("user", "assistant")
graph.add_edge("assistant", "__end__")
graph.set_entry_point("user")

# 4. Compile and run
compiled = graph.compile()

result = compiled.invoke(ChatState())

print("Messages:")
for msg in result.messages:
    print(f"  {msg['role']}: {msg['content']}")
print(f"Final step: {result.step}")
