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
| `add_fanout(source, targets, join=None)` | `Self` | Parallel fan-out execution |
| `set_entry_point(name)` | `Self` | Define start node |
| `set_finish_point(name)` | `Self` | Define terminal node |
| `set_metadata(key, value)` | `Self` | Attach metadata |
| `compile(*, input_map=None, output_map=None, checkpointer=None, name=None, state_type=None)` | `CompiledGraph` | Freeze and validate |

### `CompiledGraph[StateT]`

| Method | Returns | Description |
|---|---|---|
| `invoke(state, config=None, callbacks=None)` | `StateT` | Sync execution |
| `ainvoke(state, config=None, callbacks=None)` | `StateT` | Async execution |
| `stream(state, config=None, callbacks=None)` | `Generator[StreamEvent]` | Sync streaming |
| `astream(state, config=None, callbacks=None)` | `AsyncGenerator[StreamEvent]` | Async streaming |
| `resume(thread_id, state_type=None, updates=None, config=None, callbacks=None)` | `StateT` | Resume from last checkpoint |
| `aresume(thread_id, state_type=None, updates=None, config=None, callbacks=None)` | `StateT` | Async resume |
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

### Edge Types

| Class | Description |
|---|---|
| `DirectEdge(source, target)` | Unconditional edge |
| `ConditionalEdge(source, router, path_map)` | Conditional edge with router function |
| `ErrorEdge(source, fallback)` | Error-handling edge (fallback on failure) |
| `FanOutEdge(source, targets, join=None)` | Parallel fan-out edge |

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
EventType, StreamEvent,
Checkpoint, Checkpointer, CheckpointKey, InMemoryCheckpointer,
SqliteCheckpointer, RedisCheckpointer,
GraphExecutionPaused, ErrorEdge,
Callback, CallbackManager,
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
| Graph composition | Channels-based internals | Subgraphs and Pipelines as first-class nodes |
| Streaming | Opaque channel events | Per-step `StreamEvent` objects |
| Async support | Async API exists | Native sync+async from day 1 |
| Checkpointing | `BaseCheckpointSaver` | `Checkpointer` ABC with in-memory default |
| Callbacks | None | `Callback` Protocol + `CallbackManager` |
| Dependencies | langchain-core, pydantic, many more | Pydantic v2, typing_extensions only |
| Error handling | Let unhandled propagate | Log + dispatch to callbacks + re-raise |

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

### Future Work

 - Graph serialisation (export/import graphs as JSON or YAML)
 - A2A push notifications (webhook-based task updates)
 - OpenTelemetry tracing for node-level observability
 - Human-in-the-loop patterns (approval nodes, interrupt/resume)
 - Distributed execution (Dask/Ray integration)

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
