<div align="center">

<pre width="80">
   ________                       __     ____             __
  / ____/  _____  ____   ____ _  / /__  / __/  ____ ___  / /_
 / / __   / _ \ / __ \ / __ `/ / //_/ / /_   / __// _  |/ __/
/ /_/ /  /  __// /_/ // /_/ / / ,<   / __/ _/ /_ / ,__// /_
\____/   \___/ \___/ \__, / /_/|_| /_/   (_)___//_/    \__/
                    /____/
</pre>

**GraphForge** · _A type-safe, composable graph execution framework for LLM applications._

<br/>

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?style=flat&logo=python)](https://www.python.org)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-4A90D9?style=flat&logo=python)](https://docs.pydantic.dev)
[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat)](https://github.com/carlos07281988/GraphForge/pulls)

<br/>

</div>

**GraphForge** is a Python framework for building **stateful, composable execution graphs**
for LLM-powered applications. It takes the state-graph model popularised by
[LangGraph](https://github.com/langchain-ai/langgraph), enforces **type safety** from the
ground up, makes **state management explicit**, and keeps the **abstraction surface minimal** —
while deliberately avoiding the design mistakes of both LangGraph and LangChain.

```python
from graphforge import Graph, GraphState, node_field, Append, configure_logging

configure_logging()

class ChatState(GraphState):
    messages: list[str] = node_field(default=[], merge="append")
    next_step: str = ""

def gather(state: ChatState) -> dict:
    return {"next_step": "process"}

def process(state: ChatState) -> dict:
    return {"messages": Append(["processed"]), "next_step": "done"}

graph = (
    Graph[ChatState]()
    .add_node("gather", gather)
    .add_node("process", process)
    .add_edge("gather", "process")
    .add_edge("process", "__end__")
    .set_entry_point("gather")
    .compile()
)

result = graph.invoke(ChatState())
print(result.messages)  # ["processed"]
```

---

**其他语言**：[English](README.md) | [简体中文](README.zh-CN.md)

---

## Table of Contents

- [Why GraphForge?](#why-graphforge)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
  - [GraphState](#graphstate)
  - [Nodes](#nodes)
  - [Edges](#edges)
  - [Graph & CompiledGraph](#graph--compiledgraph)
  - [Conditional Routing](#conditional-routing)
  - [Pipeline](#pipeline)
  - [Node-level Retry & Error Fallback](#node-level-retry--error-fallback)
  - [Subgraph I/O Mapping](#subgraph-io-mapping)
  - [A2A (Agent-to-Agent) Protocol](#a2a-agent-to-agent-protocol)
  - [Agents (ToolNode + ReAct)](#agents-toolnode--react)
- [Execution Modes](#execution-modes)
  - [Invoke](#invoke)
  - [Streaming](#streaming)
  - [Async](#async)
- [Logging](#logging)
- [Checkpointing](#checkpointing)
- [Callbacks](#callbacks)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Comparison with LangGraph](#comparison-with-langgraph)
- [Contributing](#contributing)

---

## Why GraphForge?

LangGraph brought a powerful mental model — stateful graphs for agentic workflows — but
its implementation suffers from:

- **Confusing state management**: TypedDict + magic `__reducers__` that require runtime annotation parsing.
- **Weak typing**: no end-to-end type safety between nodes.
- **Opaque internals**: channel-based message passing that makes debugging hard.
- **Mutable state**: nodes can accidentally leak side effects across steps.

GraphForge takes a **third path**: keep the graph model, enforce type safety from the ground
up, make state management explicit, and minimise the surface area.

| Problem | GraphForge Solution |
|---|---|
| State schema | Pydantic v2 — validated, serialisable, introspectable |
| Merge semantics | Field-level `node_field(merge=...)` — explicit per field |
| Type safety | `Generic[StateT]` flows through every abstraction |
| Immutability | Each step = `model_copy(update=..., deep=True)` — never mutate |
| Graph composition | Subgraphs + Pipelines as first-class nodes |

---

## Installation

```bash
pip install graphforge
```

**Requires**: Python 3.9+, Pydantic v2, typing_extensions.

---

## Core Concepts

### GraphState

State is the data flowing through your graph. Define it as a Pydantic v2 `BaseModel`
and declare merge semantics per field:

```python
from graphforge import GraphState, node_field

class AgentState(GraphState):
    # Overwrite semantics (default): each node replaces the value
    status: str = "idle"

    # Append semantics: node updates are extended onto the existing list
    messages: list[dict] = node_field(default=[], merge="append")

    # Custom reducer: (old, new) -> value
    total_tokens: int = node_field(
        default=0,
        merge="reduce",
        reducer=lambda old, new: (old or 0) + new,
    )
```

**Merge strategies**:

| Strategy | Behaviour | Typical Use |
|---|---|---|
| `overwrite` | Old value replaced by new | Status flags, scalar fields |
| `append` | New items appended to list | Message history, path traces |
| `reduce` | Custom `(old, new) -> value` | Accumulators, counters |

Use `state.apply(**updates)` to produce a new snapshot:

```python
state = AgentState()
state2 = state.apply(status="running")          # overwrite
state3 = state2.apply(messages=Append([msg]))    # append
```

**States are immutable** — `apply()` always returns a new instance.

### Nodes

A node is a callable `(state, **kwargs) -> dict`. It receives the current state
and returns the fields it wants to change:

```python
def llm_call(state: AgentState) -> dict:
    response = call_llm(state.messages)
    return {
        "messages": Append([{"role": "assistant", "content": response}]),
        "total_tokens": response.tokens,
    }
```

Nodes can also be:

- **Async functions**: `async def node(state): ...`
- **Generators**: `def node(state): yield update1; yield update2`
- **CompiledGraphs**: embed a sub-graph as a node
- **Pipelines**: embed a linear pipeline as a node

### Edges

Edges connect nodes. Two types:

```python
# Unconditional edge: a always goes to b
graph.add_edge("a", "b")

# Conditional edge: router decides the target based on state
graph.add_conditional_edges(
    "classify",
    router=lambda s: "process" if s.has_data else "wait",
    path_map={"process": "processor", "wait": "input_handler"},
)
```

### Graph & CompiledGraph

Build your graph with the mutable `Graph` builder, then compile:

```python
graph = (
    Graph[AgentState]()
    .add_node("llm", llm_call)
    .add_node("tools", tool_executor)
    .add_edge("llm", "tools")
    .add_edge("tools", "__end__")      # __end__ is the terminal sentinel
    .set_entry_point("llm")
    .compile(name="my_agent")
)
```

`CompiledGraph` is immutable. Once compiled, the topology never changes.

### Conditional Routing

Routers inspect the state and return a key into `path_map`:

```python
def route_tool(state: AgentState) -> str:
    if state.needs_search:
        return "search"
    if state.needs_calc:
        return "calculator"
    return "response"

graph.add_conditional_edges(
    "router",
    router=route_tool,
    path_map={
        "search": "web_search",
        "calculator": "calc_tool",
        "response": "final_output",
    },
)
```

Return `"__end__"` from `path_map` to terminate from a conditional edge.

### Pipeline

A `Pipeline` is a linear sequence of steps (no branching). It can also be used
as a node inside a graph:

```python
from graphforge import Pipeline

pipe = Pipeline[AgentState]([
    step_one,
    step_two,
    step_three,
], name="preprocess")

# Use as a graph node
graph.add_node("preprocess", pipe)
```


### Node-level Retry & Error Fallback

Nodes can automatically retry on failure, and graphs can route to a fallback
node when a node exhausts its retries.

**Retry:** Pass ``retry=N`` to :meth:`~graphforge.Graph.add_node`:

```python
def flaky_node(state) -> dict:
    # may raise occasionally
    return {"x": 1}

graph.add_node("unstable", flaky_node, retry=3)
```

The executor retries up to ``retry + 1`` times before propagating the
exception.

**Error edge:** Use :meth:`~graphforge.Graph.add_error_edge` to define a
fallback path:

```python
def fallback(state) -> dict:
    return {"x": -1}

graph.add_node("primary", flaky_node)
graph.add_node("backup", fallback)
graph.add_error_edge("primary", "backup")
graph.add_edge("backup", "__end__")
graph.add_edge("primary", "__end__")
```

If ``primary`` raises an exception and all retries are exhausted, execution
routes to ``backup`` instead of crashing.


### Subgraph I/O Mapping

When embedding a sub-graph as a node, you can declare how the parent state
maps to and from the subgraph state with ``input_map`` and ``output_map``.

```python
class ParentState(GraphState):
    query: str = ""
    result: str = ""

class SubState(GraphState):
    prompt: str = ""
    output: str = ""

# Compile the subgraph with I/O maps
sub = (
    Graph[SubState]()
    .add_node("process", lambda s: {"output": f"Result: {s.prompt}"})
    .add_edge("process", "__end__")
    .set_entry_point("process")
    .compile(
        state_type=SubState,
        input_map={"query": "prompt"},    # parent.query -> sub.prompt
        output_map={"output": "result"},  # sub.output -> parent.result
    )
)

# Use the subgraph as a node in the parent graph
parent = Graph[ParentState]().add_node("sub", sub).add_edge("sub", "__end__").set_entry_point("sub").compile()
result = parent.invoke(ParentState(query="hello"))
print(result.result)  # "Result: hello"
```

The ``input_map`` copies fields **from** parent state **to** subgraph state;
the ``output_map`` copies fields **from** subgraph result **to** the parent
state update. Both are optional.

### A2A (Agent-to-Agent) Protocol

GraphForge includes built-in support for Google's [Agent-to-Agent (A2A)](https://google.github.io/A2A/) open protocol,
enabling agents built with GraphForge to communicate with agents built on any other framework
(and vice versa).

The A2A module provides both **outbound** and **inbound** integration:

| Direction | Mechanism | Use Case |
|---|---|---|
| **Outbound** (GraphForge → remote agent) | `create_a2a_agent_node()` | Call a third-party agent from within your graph |
| **Inbound** (remote agent → GraphForge) | `A2AServer` | Expose your graph as a standard A2A endpoint |

**Installation:**

```bash
pip install graphforge[a2a]    # adds aiohttp dependency
```

#### Outbound: Calling an External Agent

```python
from graphforge.a2a import create_a2a_agent_node

# Create a node that delegates to a remote A2A agent
call_weather = create_a2a_agent_node("http://weather-agent:8080")

# Use it like any other node
graph.add_node("get_weather", call_weather)
graph.add_edge("user_input", "get_weather")
```

Custom mappers let you control how graph state maps to A2A messages and back:

```python
def custom_input(state) -> Message:
    prompt = state.prompt or str(state)
    return Message(role="user", parts=[TextPart(text=prompt)])

def custom_output(msg, task) -> dict:
    text = msg.parts[0].text if msg and msg.parts else "done"
    return {"messages": Append([{"role": "assistant", "content": text}])}

node = create_a2a_agent_node(
    "http://agent:8080",
    input_mapper=custom_input,
    output_mapper=custom_output,
)
```

#### Inbound: Exposing a Graph as an A2A Agent

```python
from graphforge.a2a import A2AServer, AgentCard, AgentSkill

card = AgentCard(
    name="SupportBot",
    description="Customer support agent",
    capabilities=AgentCapabilities(
        skills=[AgentSkill(id="triage", name="Issue triage")],
    ),
)

server = A2AServer(
    compiled_graph,
    agent_card=card,
    host="0.0.0.0",
    port=8080,
)

# Blocking entry point
server.run()

# Or start/stop manually
await server.start()
# ... serve requests ...
await server.stop()
```

Once running, any A2A-compatible agent can discover and call your graph at:

```
GET  /.well-known/agent-card     # Discovery
POST /tasks/send                  # Synchronous task
POST /tasks/sendStream            # Streaming task (SSE)
GET  /tasks/{id}                  # Task status
POST /tasks/{id}/cancel           # Cancel task
```


---

### Agents (ToolNode + ReAct)

GraphForge ships with built-in agent patterns and tool-calling support
in the ``graphforge.agents`` module.

**ToolNode** — a node that calls an LLM, executes tool calls, and appends
results to the message list:

```python
from graphforge.agents import ToolNode

def search(query: str) -> str:
    return f"Found: {query}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Web search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        "_func": search,
    }
]

def llm(messages, tools):
    return {
        "content": None,
        "tool_calls": [{"id": "c1", "name": "search", "arguments": {"query": "hello"}}],
    }

graph.add_node("agent", ToolNode(llm, tools=tools))
```

**Router** — :func:`~graphforge.agents.has_tool_calls` checks if the last
message contains tool calls, for use with ``add_conditional_edges``:

```python
from graphforge.agents import has_tool_calls
graph.add_conditional_edges("agent", has_tool_calls, {
    "tools": "execute_tools",
    "end": "__end__",
})
```

**ReAct Agent** — :func:`~graphforge.agents.create_react_agent` builds a
complete Reasoning + Acting loop graph in one call:

```python
from graphforge.agents import create_react_agent

graph = create_react_agent(llm, tools=tools)
compiled = graph.compile(state_type=ReactState)
result = compiled.invoke(ReactState(messages=[{"role": "user", "content": "search for AI news"}]))
```




## MCP (Model Context Protocol) Integration

GraphForge supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), the industry standard for tool integration. MCP enables GraphForge agents to connect to any MCP-compatible server, auto-discover tools, and invoke them — or expose compiled graphs as MCP endpoints for other agents.

Requires ``mcp`` package (install with ``pip install graphforge[mcp]``).

### MCP Client — Call External MCP Servers

Connect to any MCP server and use its tools in your graph:

```python
from graphforge.mcp import MCPClient
from graphforge.agents import ToolNode

# Connect to an MCP server (stdio or SSE transport)
client = MCPClient("npx", args=["-y", "@modelcontextprotocol/server-filesystem"])

# Auto-discover tools and convert to ToolDef format
from graphforge.mcp import mcp_tools_to_tool_defs

async def get_tools():
    async with client:
        mcp_tools = await client.list_tools()
        tool_defs = mcp_tools_to_tool_defs(mcp_tools, client.call_tool)
        return tool_defs

# Use as a ToolNode in your graph
# ToolNode expects (llm_func, tools=tool_defs)
```

### MCP Agent Server — Expose Graph as MCP Tools

Expose each node of your compiled graph as an MCP tool:

```python
from graphforge.mcp import MCPAgentServer

server = MCPAgentServer(compiled_graph, server_name="my-agent")
server.serve()  # Starts stdio-based MCP server
```

**Transports**: Both ``stdio`` (subprocess) and ``SSE`` (HTTP) supported.

---

## Store / Long-Term Memory

Agents need persistent memory beyond execution checkpoints. The ``Store`` abstraction provides cross-session, cross-thread key-value storage for agent memory — separate from checkpoint state.

```python
from graphforge import Store, InMemoryStore

store = InMemoryStore()

store.put("session-123", "user_prefs", {"theme": "dark", "language": "zh-CN"})
prefs = store.get("session-123", "user_prefs")
print(prefs)  # {"theme": "dark", "language": "zh-CN"}

# Namespace isolation
store.put("session-456", "user_prefs", {"theme": "light"})
```

### Redis Store

For production deployments, use the Redis-backed store:

```python
from graphforge.store_redis import RedisStore

store = RedisStore(host="localhost", port=6379, db=0)
# or pass an existing Redis client:
# store = RedisStore(redis_client=existing_redis)
```

**API**:

| Method | Description |
|---|---|
| ``get(namespace, key)`` | Retrieve a value |
| ``put(namespace, key, value)`` | Store/overwrite a value |
| ``delete(namespace, key)`` | Remove a key |
| ``list_keys(namespace)`` | List all keys in a namespace |
| ``clear_namespace(namespace)`` | Remove all keys in a namespace |

---

## Multi-Agent Orchestration Patterns

GraphForge provides built-in multi-agent coordination patterns that go beyond single-agent ReAct. Each pattern is a factory function returning a compiled ``Graph`` that can be used standalone or embedded as a subgraph.

### Supervisor/Worker

A supervisor agent routes tasks to specialized workers, reviews their output, and decides when to finish:

```python
from graphforge.agents import create_supervisor_worker, SupervisorState

def supervisor_fn(state):
    task = state.get("task", "")
    if "search" in task:
        return {"current_worker": "search_worker"}
    return {"done": True, "final_answer": "No worker needed"}

def search_worker(state):
    return {"messages": [{"role": "assistant", "content": f"Searched for: {state.get('task')}"}]}

graph = create_supervisor_worker(supervisor_fn, {"search_worker": search_worker})
compiled = graph.compile(state_type=SupervisorState)
```

### Swarm

Agents hand off control to each other via a router function (OpenAI Swarm style):

```python
from graphforge.agents import create_swarm, SwarmState

def router_fn(state):
    current = state.get("current_agent", "")
    if current == "agent_a":
        return "agent_b"
    if current == "agent_b":
        return None  # terminate
    return "agent_a"

graph = create_swarm({"agent_a": agent_a_fn, "agent_b": agent_b_fn}, router_fn)
compiled = graph.compile(state_type=SwarmState)
```

### Delegation

An orchestrator agent delegates sub-tasks to specialized compiled sub-agents:

```python
from graphforge.agents import create_delegation_agent

sub_agent = Graph[TaskState]().add_node(...).compile(state_type=TaskState)
graph = create_delegation_agent(orchestrator_fn, {"sub_agent": sub_agent})
```

Each pattern comes with its own state class (``SupervisorState``, ``SwarmState``) or accepts custom state types.

---

## Guardrails (Input/Output Safety)

Guardrails provide safety validation at graph boundaries:

```python
from graphforge.guardrails import (
    InputGuardian, OutputGuardian,
    FieldLengthGuardrail,
    GuardrailError,
)

# Built-in guardrails
length_check = FieldLengthGuardrail("prompt", max_length=5000)

# Apply to graph execution
guardian = InputGuardian([length_check])
result = guardian.check({"prompt": "user input here..."})  # raises GuardrailError if blocked
```

**Custom guardrails**:

```python
from graphforge.guardrails import Guardrail, GuardrailResult

class PIIGuardrail:
    def check_input(self, state):
        text = str(state)
        if "ssn:" in text:
            return GuardrailResult.block("PII detected")
        return GuardrailResult.allow()

guardian = InputGuardian([PIIGuardrail()])
```

**Actions**: ``ALLOW`` (proceed), ``BLOCK`` (raise error), ``REPLACE`` (rewrite content).

---

## MapReduce — Parallel Data Processing

Process list-structured state fields in parallel:

```python
from graphforge._map_reduce import MapReduce

def analyze(item: str) -> str:
    return f"Processed: {item}"

def combine(results: list) -> str:
    return "\n".join(results)

# Creates a callable node for Graph.add_node()
mr = MapReduce(
    analyze, combine,
    input_field="chunks",     # State field with input list
    output_field="summary",   # State field for output
    max_workers=4,
)

graph.add_node("parallel_process", mr)
```

The map phase runs in parallel via ``ThreadPoolExecutor``; the reduce phase combines results.



## `@tool` Decorator — Auto-Generated Tool Schemas

The ``@tool`` decorator converts any Python function into a ``Tool`` with an
auto-generated OpenAI-compatible ``ToolDef``, saving you from writing JSON Schema by hand.

```python
from graphforge.tools import tool

@tool
def search(query: str, max_results: int = 10) -> str:
    """Search the web for information."""
    return f"Results for {query} (limit: {max_results})"

# Use directly
result = search(query="hello", max_results=5)

# Or with ToolNode
from graphforge.agents import ToolNode
graph.add_node("agent", ToolNode(llm, tools=[search.tool_def]))

# Inspect the generated schema
print(search.tool_def["function"]["parameters"])
# -> {"type": "object", "properties": {"query": {"type": "string"}, ...}}
```

**Features**:
- Auto-generates JSON Schema from Python type annotations
- Extracts description from docstring
- Supports optional parameters with defaults
- Supports ``Annotated`` type for field descriptions

---

## Structured Output — Pydantic-Validated LLM Outputs

Enforce that LLM outputs conform to a Pydantic model schema, with automatic
retry on validation failure.

```python
from pydantic import BaseModel
from graphforge.structured_output import with_structured_output

class Weather(BaseModel):
    city: str
    temperature: float
    conditions: str

# Wrap any LLM callable
llm = with_structured_output(my_llm_func, Weather, max_retries=3)

# Returns validated Weather instance
result = llm(messages)
print(result.city, result.temperature)

# Use as a graph node
def weather_node(state):
    weather = llm(state.messages)
    return {"weather": weather.model_dump()}
```

**Design**:
- Appends JSON schema prompt to force structured output
- Parses both raw JSON and markdown-fenced code blocks
- Retries on validation failure (configurable)
- Full type preservation via generics

---

## Agent Evaluation — Test Agent Behavior

Built-in evaluation framework for testing compiled graphs against expected outcomes:

```python
from graphforge.eval import evaluate, EvalCase, exact_match

cases = [
    EvalCase(
        input={"messages": [{"role": "user", "content": "Hello"}]},
        expected={"output": "Hi there!"},
        metrics=[exact_match("output")],
    ),
]

results = evaluate(compiled_graph, cases, state_type=ChatState)
print(results.summary())  # "EvalResults: 3/4 passed (75.0%)"

for failure in results.failures():
    print(failure.case.name, failure.metric_results)
```

**Built-in metrics**: ``exact_match``, ``contains``, ``json_match``, or custom callables.

---

## TimingCallback — Node-Level Performance Stats

Collect per-node execution timing without modifying node code:

```python
from graphforge._callbacks import TimingCallback, CallbackManager

timer = TimingCallback()
compiled.invoke(state, callbacks=CallbackManager([timer]))

for node, stats in timer.get_stats().items():
    print(f"{node}: {stats['duration']:.3f}s ({stats['calls']} calls)")
```

---

## Cancellation API — Stop Running Graphs

Cancel a long-running graph execution from another thread:

```python
import threading

def run_in_thread(compiled, state):
    result = compiled.invoke(state, config={"thread_id": "my-task"})
    return result

thread = threading.Thread(target=run_in_thread, args=(compiled, state))
thread.start()

# Cancel from main thread
compiled.cancel("my-task")
thread.join()
```

When cancelled, the graph saves a checkpoint of the current state before
raising ``GraphExecutionPaused``.

---


## Token-Level Streaming (Generator Nodes)

Nodes can now be generator functions that ``yield`` intermediate state updates, enabling
token-by-token LLM output streaming (typewriter effect):

```python
def token_generator(state):
    """Generator node that yields tokens one by one."""
    for token in ["Hello", " ", "World"]:
        yield {"output": token}

graph.add_node("stream_gen", token_generator)

# In stream mode, each yield becomes a STREAM_TOKEN event
for event in compiled.stream(state, stream_mode="events"):
    if event.type == EventType.STREAM_TOKEN:
        print(event.data)  # {"token": {"output": "Hello"}}, etc.
```

Generator nodes work in both ``invoke()`` (collects all yields) and ``stream()``
(emits individual ``STREAM_TOKEN`` events per yield).

---

## WebSocket Streaming Endpoint

The ``GraphServer`` now exposes a WebSocket endpoint for bidirectional real-time
communication:

```python
from graphforge._http_server import GraphServer

server = GraphServer(compiled_graph, host="0.0.0.0", port=8080)
server.run()  # Now also supports ws://host:port/ws
```

Connect via WebSocket:
```json
// Send: {"state": {...}, "config": {...}}
// Receive: {"type": "node_start", "node": "a", "data": "..."}
```

---

## Node-Level Checkpoint Skipping

Pure function nodes can skip checkpointing for performance:

```python
# 'transform' will NOT be checkpointed
graph.add_node("transform", clean_data, checkpoint=False)

# Default is True (backward compatible)
graph.add_node("llm_call", call_llm, checkpoint=True)
```

---

## Configuration System (Configurable Fields)

Override state field values at invocation time without recompiling:

```python
result = compiled.invoke(
    MyState(),
    configurable={"model": "gpt-4", "temperature": 0.9},
)
```

Configurable values are merged into the state before the first node executes.

---

## Background Task Execution

Run graphs in background threads with status tracking:

```python
from graphforge.background import BackgroundTaskRunner

runner = BackgroundTaskRunner(max_workers=4)
task = runner.submit(compiled_graph, input_state)

# Check status, wait for result
task.wait(timeout=30)
print(task.status)  # "completed" | "failed"
print(task.result)

# List all tasks
for t in runner.list_tasks():
    print(t.task_id, t.status)
```

---

## CLI Tool

Basic command-line interface for common operations:

```bash
python -m graphforge.cli info graph.json      # Show topology
python -m graphforge.cli viz graph.json       # Export visualization
python -m graphforge.cli run graph.json state.json  # Run graph
```

---

## State Middleware

Pre- and post-processing hooks for state transitions:

```python
from graphforge._middleware import MiddlewarePipeline, StateMiddleware

class LoggingMiddleware:
    def pre_update(self, node, state, updates):
        print(f"{node}: updating {list(updates.keys())}")
        return updates

    def post_update(self, node, old_state, new_state):
        print(f"{node}: state changed")

pipeline = MiddlewarePipeline([LoggingMiddleware()])
compiled.invoke(state, callbacks=CallbackManager([], middleware=pipeline))
```

---



## RAG (Retrieval-Augmented Generation)

Built-in module for knowledge retrieval with embeddings, vector stores, and
retrieval nodes.

```python
from graphforge.rag import InMemoryVectorStore, RetrievalNode, chunk_text
from graphforge.rag._embeddings import DeterministicEmbeddings

# Create embeddings and vector store
embeddings = DeterministicEmbeddings(dimension=128)
store = InMemoryVectorStore(embeddings)

# Add documents (auto-chunked and embedded)
store.add_texts([
    "Paris is the capital of France",
    "Python is a programming language",
])

# Use in graph as retrieval node
graph.add_node("retrieve", RetrievalNode(store, query_field="question"))

# Custom embeddings for production
# from graphforge.rag import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

**Chunking utilities** for splitting text before embedding:

```python
from graphforge.rag import chunk_text, ChunkingStrategy

chunks = chunk_text(long_text, strategy="recursive", chunk_size=500)
```

---

## Cost / Token Tracking

Track token usage and estimated costs per node via the callback system:

```python
from graphforge import CallbackManager
from graphforge._callbacks import CostCallback

cost = CostCallback()
cm = CallbackManager([cost])
compiled.invoke(state, callbacks=cm)

print(f"Total cost: ${cost.total_cost():.4f}")
print(f"Total tokens: {cost.total_tokens()}")

for node, stats in cost.get_stats().items():
    print(f"{node}: {stats['cost']:.4f} ({stats['total_tokens']} tokens)")

# Custom pricing
cost.set_pricing("my-model", input_price=0.01, output_price=0.03)

# Track usage manually
cost.track("gpt-4", prompt_tokens=100, completion_tokens=50, node="llm_call")
```

**Built-in pricing** for GPT-4, GPT-4o, GPT-4o-mini, GPT-3.5, Claude 3 models.

---

## Human-in-the-Loop Approval Patterns

The ``interrupt()`` function now supports configurable timeouts:

```python
from graphforge import interrupt

def review_node(state):
    response = interrupt(
        message="Approve this action?",
        value={"action": "send_email"},
        timeout=3600,       # 1 hour timeout
        on_timeout="reject",  # auto-reject on timeout
    )
    # Resume with updates={"decision": "approve"}
    return {"approved": response.get("decision") == "approve"}
```

**ApprovalNode** wraps any function with an approval gate:

```python
from graphforge.agents import ApprovalNode

def send_email(state):
    # Send email logic
    return {"sent": True}

# Wrapped: pauses for human approval before executing
graph.add_node("send_email", ApprovalNode(send_email, timeout=300))
```

When interrupted, the graph saves a checkpoint. Resume with
``compiled.resume(thread_id, updates={"decision": "approve"})``.

---



## `serve()` — One-Command API Server

Turn any compiled graph into a production-ready API server with a single function call.
Starts REST, WebSocket, MCP, and A2A servers simultaneously:

```python
from graphforge import serve

# One line → REST + WebSocket + MCP + A2A on port 8080
serve(compiled_graph)

# Custom port + auth
serve(graph, port=9090, api_key="sk-...")
```

**Endpoints**: ``POST /invoke``, ``POST /stream`` (SSE), ``GET /ws`` (WebSocket),
``GET /health``, ``GET /docs`` (auto-generated Swagger).

---

## AutoOptimizer — Automatic Graph Parallelization

Static analysis that detects independent execution paths and suggests or
automatically applies parallelization:

```python
from graphforge._optimizer import optimize, auto_parallelize

# Analyze an existing compiled graph
report = optimize(compiled_graph)
print(report.summary())
# → Independent paths: 2, Unused nodes: 0, Bottlenecks: 1

# Apply suggestions to a graph builder
optimized = auto_parallelize(graph_builder, state_type)
# Independent paths are now parallel fan-out edges
```

The optimizer detects: parallelizable paths, unused nodes, bottlenecks, cycles.

---

## TimelineRecorder — Execution Debugger

Record every state transition for post-mortem debugging and replay:

```python
from graphforge import CallbackManager
from graphforge._timeline import TimelineRecorder

recorder = TimelineRecorder()
compiled.invoke(state, callbacks=CallbackManager([recorder]))

# Inspect
for frame in recorder.get_timeline():
    print(f"{frame.node}: {list(frame.updates.keys())} ({frame.duration:.3f}s)")

# Export to JSON
recorder.export_json("trace.json")

# Replay step by step
for frame in recorder.replay():
    print(frame.node, frame.state_after)
```

Each frame captures: node name, step, state before/after, updates, duration.


## Execution Modes

### Invoke

```python
result = compiled.invoke(initial_state, config={
    "thread_id": "session-123",
    "recursion_limit": 50,
})
```

### Streaming

Streaming yields per-step events for live UIs or progress monitoring:

```python
from graphforge import EventType

for event in compiled.stream(initial_state):
    if event.type == EventType.NODE_START:
        print(f"⏳ {event.node}")
    elif event.type == EventType.STATE_UPDATE:
        print(f"  ├ updates: {list(event.data['updates'].keys())}")
    elif event.type == EventType.NODE_END:
        print(f"  └ state: {event.data}")
    elif event.type == EventType.GRAPH_END:
        print(f"✅ done: {event.data}")
```

Event sequence for a 3-node graph:

```
GRAPH_START  {graph: "my_graph"}
  NODE_START   {node: "a", state: {...}}     step 0
  STATE_UPDATE {node: "a", updates: {...}}   step 0
  NODE_END     {node: "a", state: {...}}     step 0
  CONDITIONAL  {node: "a", next: "b"}        step 0
  NODE_START   {node: "b", state: {...}}     step 1
  STATE_UPDATE {node: "b", updates: {...}}   step 1
  NODE_END     {node: "b", state: {...}}     step 1
GRAPH_END    {state: {...}}
```

### Async

```python
import asyncio

# Async invoke
result = asyncio.run(compiled.ainvoke(initial_state))

# Async streaming
async for event in compiled.astream(initial_state):
    print(event)
```

The `AsyncExecutor` handles both sync and async nodes transparently — sync nodes
are invoked normally, async nodes are awaited.

---

## Logging

GraphForge includes structured logging across all modules.
The only import-time output is a single `DEBUG` message.

### Quick Setup

```python
from graphforge import configure_logging
import logging

configure_logging(level=logging.DEBUG)
```

This outputs:

```
09:15:42 [graphforge] INFO GraphForge logging configured at level DEBUG
09:15:42 [graphforge.graph] INFO Compiling graph: 3 nodes, 4 edges, 1 conditional edges
09:15:42 [graphforge.executor] INFO Graph "my_agent" starting (thread="default", entry="classify", recursion_limit=100)
09:15:42 [graphforge.executor] INFO Node "classify" (step=0, kind=function)
09:15:42 [graphforge.executor] INFO Node "process" (step=1, kind=function)
09:15:42 [graphforge.executor] INFO Graph "my_agent" finished in 2 steps
```

### Module Loggers

All loggers live under the `graphforge` namespace:

| Logger | Module | Typical Level |
|---|---|---|
| `graphforge.graph` | Graph builder & compilation | INFO, DEBUG |
| `graphforge.node` | Node dispatch | DEBUG |
| `graphforge.state` | State merge | DEBUG |
| `graphforge.executor` | Execution engine | INFO, DEBUG, ERROR |
| `graphforge.checkpoint` | Checkpointing | DEBUG |
| `graphforge.callback` | Callback dispatch | DEBUG |
| `graphforge.pipeline` | Pipeline execution | INFO, DEBUG |

### Custom Logging

You can also configure Python's standard logging directly:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
```

---

## Checkpointing

Checkpoints save state after each node, enabling resumption and debugging:

```python
from graphforge import InMemoryCheckpointer

checkpointer = InMemoryCheckpointer()
compiled = graph.compile(checkpointer=checkpointer)
compiled.invoke(initial_state)

# Inspect execution history
for key in checkpointer.list("default"):
    snapshot = checkpointer.get(key)
    print(f"  {key[0]} / {key[1]} / step {key[2]} -> {snapshot.state}")
```

To implement a custom backend (Redis, Postgres, S3), subclass `Checkpointer`:

```python
from graphforge import Checkpointer, Checkpoint, CheckpointKey
from typing import Optional, List, Dict, Any

class MyCheckpointer(Checkpointer[MyState]):
    def put(self, key: CheckpointKey, state: Dict[str, Any],
            parent_key: Optional[CheckpointKey] = None) -> None: ...
    def get(self, key: CheckpointKey) -> Optional[Checkpoint[MyState]]: ...
    def list(self, thread_id: str) -> List[CheckpointKey]: ...
```

---

## Callbacks

Callbacks let you hook into execution without modifying nodes:

```python
from graphforge import Callback, CallbackManager

class Logger(Callback):
    def on_node_start(self, node: str, state: dict) -> None:
        print(f"  starting {node}")

    def on_node_end(self, node: str, state: dict) -> None:
        print(f"  finished {node}")

    def on_node_error(self, node: str, error: Exception) -> None:
        print(f"  ERROR in {node}: {error}")

manager = CallbackManager([Logger()])
compiled.invoke(state, callbacks=manager)
```

**Available hooks**:

| Hook | Called When |
|---|---|
| `on_graph_start(graph_name, input_state)` | Execution begins |
| `on_graph_end(graph_name, final_state)` | Execution completes |
| `on_graph_error(graph_name, error)` | Unhandled exception |
| `on_node_start(node, state)` | Just before node runs |
| `on_node_end(node, state)` | Just after node succeeds |
| `on_node_error(node, error)` | Node raises an exception |
| `on_state_update(node, updates, new_state)` | State is merged |
| `on_conditional_edge(node, result, target)` | Router is evaluated |

---

## API Reference

### `Graph[StateT]`

| Method | Returns | Description |
|---|---|---|
| `add_node(name, fn, *, retry=0, timeout=None, metadata=None)` | `Self` | Register a node with optional retry |
| `add_edge(source, target)` | `Self` | Unconditional edge |
| `add_error_edge(source, fallback)` | `Self` | Route to fallback node on error |
| `add_conditional_edges(source, router, path_map)` | `Self` | Conditional routing |
| `add_fanout(source, targets, join=None, *, conflict=None)` | `Self` | Parallel fan-out with optional conflict strategy |
| `add_sequence([a, b, c])` | `Self` | Linear chain of nodes (sequential edges) |
| `add_parallel(source, [a, b], join=j, *, conflict=None)` | `Self` | Parallel fan-out from source to targets |
| `set_entry_point(name)` | `Self` | Define start node |
| `set_finish_point(name)` | `Self` | Define terminal node |
| `set_metadata(key, value)` | `Self` | Attach metadata |
| `compile(*, input_map=None, output_map=None, checkpointer=None, name=None, state_type=None, stream_modes=None)` | `CompiledGraph` | Freeze and validate |

### `CompiledGraph[StateT]`

| Method | Returns | Description |
|---|---|---|
| `invoke(state, config=None, callbacks=None, *, store=None)` | `StateT` | Sync execution with optional store |
| `ainvoke(state, config=None, callbacks=None, *, store=None)` | `StateT` | Async execution with optional store |
| `stream(state, config=None, callbacks=None, *, store=None, stream_mode="events")` | `Generator[StreamEvent]` | Sync streaming (modes: events/values/updates/debug) |
| `astream(state, config=None, callbacks=None, *, store=None, stream_mode="events")` | `AsyncGenerator[StreamEvent]` | Async streaming |
| `resume(thread_id, state_type=None, updates=None, config=None, callbacks=None, *, store=None)` | `StateT` | Resume from last checkpoint |
| `aresume(thread_id, state_type=None, updates=None, config=None, callbacks=None, *, store=None)` | `StateT` | Async resume |
| `cancel(thread_id)` | `None` | Cancel a running execution |
| `clear_cancel(thread_id)` | `None` | Clear cancellation signal |
| Properties: `name`, `nodes`, `entry_point`, `finish_points`, `checkpointer`, `metadata`, `state_type`, `input_map`, `output_map`, `error_map` |  | Read-only |

### `GraphState`

| Method | Returns | Description |
|---|---|---|
| `apply(**updates)` | `Self` | Immutable merge of field updates |


### `Node`

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Node identifier |
| `kind` | `NodeKind` | Runtime classification (function, async, subgraph, etc.) |
| `retry` | `int` | Number of retry attempts on failure |
| `timeout` | `float | None` | Maximum execution time in seconds |
| `metadata` | `dict` | Optional user-defined metadata |

### Agents

| Function | Description |
|---|---|
| `ToolNode(llm_func, tools, state_messages_field="messages")` | Create a tool-calling agent node |
| `has_tool_calls(state, field="messages")` | Conditional router: returns `"tools"` or `"end"` |
| `create_react_agent(llm_func, tools, state_type=None)` | Build a ReAct agent graph |
| `create_supervisor_worker(supervisor_fn, workers, max_iterations=10)` | Supervisor/Worker pattern |
| `create_swarm(agents, router_fn)` | Swarm pattern |
| `create_delegation_agent(orchestrator_fn, sub_agents)` | Delegation pattern |

### Edge Types

| Class | Description |
|---|---|
| `DirectEdge(source, target)` | Unconditional edge |
| `ConditionalEdge(source, router, path_map)` | Conditional edge with router function |
| `FanOutEdge(source, targets, join)` | Parallel fan-out edge |

### `Pipeline[StateT]`

| Method | Returns | Description |
|---|---|---|
| `run(state, **kwargs)` | `StateUpdate` | Sync sequential execution |
| `arun(state, **kwargs)` | `StateUpdate` | Async sequential execution |

### Utility Functions

- `node_field(default=None, merge="overwrite", reducer=None, description="", **extra)` — Declare a state field with merge behaviour.
- `configure_logging(level=INFO, fmt=..., datefmt=...)` — Configure framework logging.
- `get_logger(name)` — Get a `graphforge.*` child logger.

### Types

- `NodeName = str`
- `StateUpdate = dict[str, Any]`
- `ConfigDict = dict[str, Any]`
- `NodeFunc = Callable[..., StateUpdate]`
- `AsyncNodeFunc = Callable[..., Awaitable[StateUpdate]]`
- `RouterFunc = Callable[..., NodeName]`
- `AsyncRouterFunc = Callable[..., Awaitable[NodeName]]`

### Symbols

```
Graph, CompiledGraph, Node, NodeKind,
GraphState, Append, MergeStrategy, node_field,
Pipeline,
EventType, StreamEvent, StreamMode,
Checkpoint, Checkpointer, CheckpointKey, InMemoryCheckpointer,
SqliteCheckpointer, RedisCheckpointer, PostgresCheckpointer,
GraphExecutionPaused,
Callback, CallbackManager, TimingCallback,
Store, InMemoryStore, RedisStore,
Guardrail, GuardrailAction, GuardrailError, GuardrailResult,
InputGuardian, OutputGuardian, FieldLengthGuardrail,
MapReduce,
Tool, tool, StructuredOutputWrapper, with_structured_output,
EvalCase, EvalResults, evaluate, exact_match, contains, json_match,
MCPClient, MCPAgentServer, mcp_tools_to_tool_defs,
configure_logging, get_logger,
export_dot, render_graph,
NodeFunc, AsyncNodeFunc, RouterFunc, AsyncRouterFunc,
StreamingNodeFunc, AsyncStreamingNodeFunc,
NodeName, StateUpdate, ConfigDict
```

---

## Architecture

For a deep dive into the design, see:

[`docs/architecture.md`](docs/architecture.md) · [`docs/architecture.zh-CN.md`](docs/architecture.zh-CN.md)（中文版）

Topics covered:

- Module dependency graph
- State merge pipeline
- Execution lifecycle (sync, async, streaming)
- Checkpointing model
- Design decisions & rationale
- Extension points
- Full comparison with LangGraph and LangChain

---

## Comparison with LangGraph

| Area | LangGraph | GraphForge |
|---|---|---|
| State model | TypedDict + magic `__reducers__` | Pydantic v2 with explicit `node_field(merge=...)` |
| Type safety | Weak (TypedDict is runtime-fragile) | Full generics, Protocols, static analysis |
| Immutability | Mutable state in place | Each step = new snapshot with `model_copy` |
| Graph composition | Channels-based internals | Subgraphs, Pipelines, add_sequence/add_parallel |
| Streaming | Opaque channel events | Per-step `StreamEvent` + values/updates/debug modes |
| Async support | Async API exists | Native sync+async from day 1 |
| Checkpointing | `BaseCheckpointSaver` | `Checkpointer` ABC + InMemory/SQLite/Redis/Postgres |
| Callbacks | None | `Callback` Protocol + `CallbackManager` + `TimingCallback` |
| MCP | Native MCP tool integration | `MCPClient` + `MCPAgentServer` |
| Store / Memory | `BaseStore` cross-thread KV | `Store` ABC + `InMemoryStore` + `RedisStore` |
| Guardrails | Built-in guardrails | `InputGuardian` + `OutputGuardian` + custom `Guardrail` |
| Multi-Agent | Supervisor, Swarm, Map-Reduce | Supervisor/Worker, Swarm, Delegation, ReAct |
| Tool definitions | `@tool` decorator | `@tool` decorator + auto JSON Schema |
| Structured Output | `with_structured_output` | `with_structured_output` + retry + validation |
| Cancellation | Task cancellation | `cancel(thread_id)` via threading.Event |
| Agent Evaluation | LangSmith integration | Built-in `EvalCase` + `evaluate()` + metrics |
| Dependencies | langchain-core, pydantic, many more | Pydantic v2, typing_extensions only |
| Error handling | Let unhandled propagate | Log + dispatch + retry + error edges |

---

## Contributing

```bash
# Clone and install
git clone https://github.com/carlos07281988/GraphForge.git
cd GraphForge
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Check types (if mypy is available)
mypy graphforge/
```

The framework currently targets **70+ tests** across state management, graph building,
execution, streaming, checkpointing, callbacks, and pipelines.

A detailed record of all improvements is maintained in [](docs/improvements.md).

### Roadmap

- [x] `resume()` API for long-running agents — checkpoint-based resumption with pause/retry support
- [x] SQLite checkpointer — persistent state storage with full CRUD, metadata, and thread safety
- [x] Parallel / fan-out node execution — add_fanout() API, async parallel via asyncio.gather, sync sequential, join support
- [x] Subgraph checkpoint isolation — automatic thread_id prefix for nested graphs, shared checkpointer support
- [x] Graph visualisation — export_dot() to DOT format, render_graph() to image
- [x] Redis checkpointer — distributed state persistence via Redis
- [x] Pydantic v1 compatibility — unified compat layer for v1/v2 APIs
- [x] A2A (Agent-to-Agent) protocol — outbound/inbound agent communication
- [x] Node-level retry & error fallback — retry=N, add_error_edge()
- [x] Subgraph I/O mapping — input_map/output_map for clean parent/child boundaries
- [x] Agents module — ToolNode, has_tool_calls(), create_react_agent()
- [x] Graph serialisation — serialize()/deserialize() for JSON/YAML export/import

### Future Work

 - A2A push notifications (webhook-based task updates)
 - OpenTelemetry tracing for node-level observability (tracing callback exists, needs deeper integration)
 - Human-in-the-loop patterns (approval nodes, interrupt/resume)
 - Distributed execution (Dask/Ray integration)
 - Persistent Store backends (Postgres, S3, file-based)
 - Streaming mode variants (values/updates/debug/custom)
 - Dynamic graph — runtime node addition/removal
 - Agent evaluation framework for testing agent behavior
 - WebSocket streaming for real-time chat

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
