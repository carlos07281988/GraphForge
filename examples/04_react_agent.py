"""
Example 04: ReAct Agent
A complete Reasoning + Acting agent loop with tool calling.
"""
from graphforge import GraphState, node_field, Append, configure_logging
from graphforge.agents import ToolNode, has_tool_calls, create_react_agent, ReactState
from graphforge.tools import tool
from graphforge._types import END_SENTINEL

configure_logging()


# Define tools using the @tool decorator
@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for '{query}': found 3 relevant documents."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


tools = [search.tool_def, calculator.tool_def]


# Define the LLM function (simulated — replace with real API call)
def llm_func(messages, tool_defs):
    """Simulate an LLM that calls tools or responds directly."""
    last = messages[-1] if messages else {}
    content = last.get("content", "").lower()

    if "search" in content:
        return {
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "name": "search",
                "arguments": {"query": content.replace("search", "").strip()},
            }],
        }
    elif "calculate" in content or any(c in content for c in "+-*/"):
        return {
            "content": None,
            "tool_calls": [{
                "id": "call_2",
                "name": "calculator",
                "arguments": {"expression": content.replace("calculate", "").strip()},
            }],
        }
    else:
        return {"content": f"I processed your request: {content}", "tool_calls": []}


# Method 1: Build ReAct agent manually
print("=== Manual ReAct Agent ===")
manual_graph = Graph[ReactState]()
manual_graph.add_node("agent", ToolNode(llm_func, tools=tools))
manual_graph.add_node("execute_tools", ToolNode(llm_func, tools=tools))
manual_graph.add_conditional_edges(
    "agent", has_tool_calls,
    {"tools": "execute_tools", "end": END_SENTINEL},
)
manual_graph.add_edge("execute_tools", "agent")
manual_graph.set_metadata("agent_type", "react")
manual_graph.set_entry_point("agent")

manual_compiled = manual_graph.compile(state_type=ReactState)
result = manual_compiled.invoke(ReactState(messages=[{"role": "user", "content": "search for AI news"}]))
print("Final messages:")
for msg in result.messages:
    role = msg.get("role", "?")
    content = msg.get("content", "")
    tool_calls = msg.get("tool_calls", [])
    if content:
        print(f"  [{role}]: {content[:60]}")
    if tool_calls:
        for tc in tool_calls:
            print(f"  [{role} -> tool]: {tc.get('name', '?')}")

# Method 2: Use create_react_agent helper
print("\n=== create_react_agent Helper ===")
react_graph = create_react_agent(llm_func, tools=tools)
react_compiled = react_graph.compile(state_type=ReactState)
result2 = react_compiled.invoke(ReactState(messages=[{"role": "user", "content": "calculate 2 + 2"}]))
for msg in result2.messages:
    role = msg.get("role", "?")
    content = msg.get("content", "")
    tc = msg.get("tool_calls", [])
    if content:
        print(f"  [{role}]: {content[:60]}")
    if tc:
        for t in tc:
            print(f"  [{role} -> tool]: {t.get('name', '?')}")
