"""
Example 03: Streaming
Demonstrates streaming modes and token-level generator nodes.
"""
from graphforge import Graph, GraphState, node_field, EventType, StreamMode, configure_logging
import time

configure_logging()


class StreamState(GraphState):
    output: str = ""
    tokens: list = node_field(default=[], merge="append")


# Generator node — yields tokens one by one
def token_generator(state: StreamState):
    """Simulate token-by-token LLM output."""
    text = "Hello! This is a streaming response from GraphForge."
    for word in text.split(" "):
        yield {"output": word, "tokens": [word]}
        time.sleep(0.05)  # simulate generation delay


# Normal node — returns complete result
def finalize(state: StreamState) -> dict:
    return {"output": state.output + " [complete]"}


# Build graph
graph = Graph[StreamState]()
graph.add_node("generate", token_generator)
graph.add_node("finalize", finalize)
graph.add_edge("generate", "finalize")
graph.add_edge("finalize", "__end__")
graph.set_entry_point("generate")

compiled = graph.compile()

# Option 1: Stream with events (token-level)
print("=== Event Streaming (token-level) ===")
for event in compiled.stream(StreamState(), stream_mode="events"):
    if event.type == EventType.STREAM_TOKEN:
        print(f"  Token: {event.data['token']['output']}", flush=True)
    elif event.type == EventType.NODE_START:
        print(f"  Starting: {event.node}")
    elif event.type == EventType.NODE_END:
        print(f"  Finished: {event.node}")
    elif event.type == EventType.GRAPH_END:
        print(f"  Done: {event.data}")

# Option 2: Stream with values mode (full state after each step)
print("\n=== Value Streaming (state snapshots) ===")
for event in compiled.stream(StreamState(), stream_mode="values"):
    if hasattr(event, 'data') and event.data:
        print(f"  State output: {event.data.get('output', '')[:50]}...")

# Option 3: Normal invoke (aggregates all tokens)
print("\n=== Normal Invoke (aggregated) ===")
result = compiled.invoke(StreamState())
print(f"  Final output: {result.output}")
print(f"  All tokens: {result.tokens}")
