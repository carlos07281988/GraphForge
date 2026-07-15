"""
Example 02: Conditional Routing
Use conditional edges for dynamic execution paths based on state.
"""
from graphforge import Graph, GraphState, node_field, Append, configure_logging

configure_logging()


class RoutingState(GraphState):
    input_text: str = ""
    category: str = ""
    result: str = ""
    path: list = node_field(default=[], merge="append")


def classifier(state: RoutingState) -> dict:
    """Classify the input text into a category."""
    text = state.input_text.lower()
    if "?" in text:
        category = "question"
    elif "!" in text:
        category = "exclamation"
    elif "help" in text or "urgent" in text:
        category = "support"
    else:
        category = "general"
    return {"category": category, "path": Append([f"classified as {category}"])}


def handle_question(state: RoutingState) -> dict:
    return {"result": f"Answering: {state.input_text}", "path": Append(["question_handler"])}


def handle_support(state: RoutingState) -> dict:
    return {"result": f"Support ticket created for: {state.input_text}", "path": Append(["support_handler"])}


def handle_general(state: RoutingState) -> dict:
    return {"result": f"General response to: {state.input_text}", "path": Append(["general_handler"])}


def router(state: RoutingState) -> str:
    """Route based on category."""
    return state.category


graph = Graph[RoutingState]()
graph.add_node("classifier", classifier)
graph.add_node("question", handle_question)
graph.add_node("support", handle_support)
graph.add_node("general", handle_general)

graph.add_conditional_edges(
    "classifier",
    router=router,
    path_map={
        "question": "question",
        "support": "support",
        "exclamation": "question",
        "general": "general",
    },
)
graph.add_edge("question", "__end__")
graph.add_edge("support", "__end__")
graph.add_edge("general", "__end__")
graph.set_entry_point("classifier")

compiled = graph.compile()

for text in ["Can you help me?", "This is urgent!", "Just a normal message"]:
    result = compiled.invoke(RoutingState(input_text=text))
    print(f"\nInput: {text!r}")
    print(f"  Category: {result.category}")
    print(f"  Result:   {result.result}")
    print(f"  Path:     {' → '.join(result.path)}")
