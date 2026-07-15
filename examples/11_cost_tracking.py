"""
Example 11: Cost Tracking
Track token usage and estimate costs for graph executions.
"""
from graphforge import Graph, GraphState, node_field, Append, CallbackManager, configure_logging
from graphforge._callbacks import CostCallback

configure_logging()


class CostState(GraphState):
    messages: list = node_field(default=[], merge="append")
    result: str = ""


def llm_call_1(state: CostState) -> dict:
    """Simulate an LLM call with token usage."""
    return {
        "messages": Append([{"role": "assistant", "content": "First response"}]),
        "result": "intermediate",
    }


def llm_call_2(state: CostState) -> dict:
    """Another LLM call with different token counts."""
    return {
        "messages": Append([{"role": "assistant", "content": "Final response"}]),
        "result": "final",
    }


# Build graph
graph = Graph[CostState]()
graph.add_node("llm1", llm_call_1)
graph.add_node("llm2", llm_call_2)
graph.add_edge("llm1", "llm2")
graph.add_edge("llm2", "__end__")
graph.set_entry_point("llm1")
compiled = graph.compile(state_type=CostState)

# Track costs
cost = CostCallback()
cm = CallbackManager([cost])

# Execute, recording costs manually (in a real app, the LLM client would do this)
result = compiled.invoke(CostState(), callbacks=cm)
cost.track("gpt-4", prompt_tokens=150, completion_tokens=50, node="llm1")
cost.track("gpt-4", prompt_tokens=100, completion_tokens=200, node="llm2")

# Report
print("=== Cost Report ===")
print(f"{'Node':<10} {'Tokens':<10} {'Cost':<10}")
print("-" * 30)
for node, stats in cost.get_stats().items():
    print(f"{node:<10} {stats['total_tokens']:<10} ${stats['cost']:<8.4f}")
print("-" * 30)
print(f"{'TOTAL':<10} {cost.total_tokens():<10} ${cost.total_cost():<8.4f}")

# Custom pricing
print("\n=== Custom Model Pricing ===")
cost.set_pricing("my-custom-model", input_price=0.005, output_price=0.015)
cost.track("my-custom-model", prompt_tokens=500, completion_tokens=300, node="custom_call")
stats = cost.get_stats().get("custom_call", {})
print(f"Custom model cost: ${stats.get('cost', 0):.4f}")

# Reset and start fresh
cost.reset()
print(f"\nAfter reset — total cost: ${cost.total_cost():.4f}")
