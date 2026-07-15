"""
Example 09: Serve
Deploy a compiled graph as a unified API server.
"""
from graphforge import Graph, GraphState, Append, node_field, serve, configure_logging
import time

configure_logging()

"""
This example shows how to deploy a graph as an API server using serve().

NOTE: Running this file starts a server on port 8080. Press Ctrl+C to stop.

Once started, you can test it with:
  curl -X POST http://localhost:8080/invoke \\
    -H "Content-Type: application/json" \\
    -d '{"state": {"messages": [{"role":"user","content":"hello"}]}}'
"""


class ChatState(GraphState):
    messages: list = node_field(default=[], merge="append")
    response: str = ""


def echo_node(state: ChatState) -> dict:
    last = state.messages[-1] if state.messages else {}
    content = last.get("content", "")
    return {
        "response": f"Echo: {content}",
        "messages": Append([{"role": "assistant", "content": f"You said: {content}"}]),
    }


# Build and compile
graph = Graph[ChatState]()
graph.add_node("echo", echo_node)
graph.add_edge("echo", "__end__")
graph.set_entry_point("echo")
compiled = graph.compile()

# Test locally first
result = compiled.invoke(ChatState(messages=[{"role": "user", "content": "Hello!"}]))
print(f"Local test: {result.response}")

# Start API server
# Uncomment the line below to start the server:
# serve(compiled, host="0.0.0.0", port=8080)

# To run this example:
#   python examples/09_serve.py
# Then in another terminal:
#   curl -X POST http://localhost:8080/invoke -H "Content-Type: application/json" \
#     -d '{"state": {"messages": [{"role":"user","content":"hello"}]}}'
print("\nTo start the server:")
print("  Uncomment serve() call in this file and run:")
print("  python examples/09_serve.py")
print("\nThen test with:")
print("  curl -X POST http://localhost:8080/invoke \\")
print('    -H "Content-Type: application/json" \\')
print('    -d \'{"state": {"messages": [{"role":"user","content":"hello"}]}}\'')
