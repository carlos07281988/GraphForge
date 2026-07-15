<div align="center">

<pre width="80">
   ______                 __    ______                    
  / ____/________ _____  / /_  / ____/___  _________ ____ 
 / / __/ ___/ __ `/ __ \/ __ \/ /_  / __ \/ ___/ __ `/ _ \
/ /_/ / /  / /_/ / /_/ / / / / __/ / /_/ / /  / /_/ /  __/
\____/_/   \__,_/ .___/_/ /_/_/    \____/_/   \__, /\___/ 
               /_/                           /____/       
</pre>

**GraphForge** · _A type-safe, composable graph execution framework for LLM applications._  
_Inspired by LangGraph, engineered to avoid its mistakes._

<br/>

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?style=flat&logo=python)](https://www.python.org)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-4A90D9?style=flat&logo=python)](https://docs.pydantic.dev)
[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat)](https://github.com/carlos07281988/GraphForge/pulls)

<br/>

</div>

```python
from graphforge import Graph, GraphState, node_field, Append

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

## Table of Contents

- [Quick Start](#quick-start)
- [Why GraphForge?](#why-graphforge)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Agent Features](#agent-features)
- [Streaming & Execution](#streaming--execution)
- [Persistence & Memory](#persistence--memory)
- [Safety & Control](#safety--control)
- [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
- [Developer Tools](#developer-tools)
- [Deployment](#deployment)
- [Observability](#observability)
- [Comparison with LangGraph](#comparison-with-langgraph)
- [API Reference](#api-reference)
- [Contributing](#contributing)

---

## Quick Start

```python
from graphforge import Graph, GraphState, node_field, Append, configure_logging

configure_logging()

class AgentState(GraphState):
    messages: list = node_field(default=[], merge="append")
    tokens: int = 0

def agent(state: AgentState) -> dict:
    return {
        "messages": Append([{"role": "assistant", "content": "Hello!"}]),
        "tokens": 42,
    }

graph = Graph[AgentState]()
graph.add_node("agent", agent)
graph.add_edge("agent", "__end__")
graph.set_entry_point("agent")
compiled = graph.compile()

result = compiled.invoke(AgentState())
print(result.messages)  # [{"role": "assistant", "content": "Hello!"}]
```

---

## Why GraphForge?

| Problem | LangGraph | GraphForge |
|---|---|---|
| State model | TypedDict + magic `__reducers__` | Pydantic v2 with explicit `node_field(merge=...)` |
| Type safety | Weak (TypedDict is runtime-fragile) | Full generics, Protocols, static analysis |
| Immutability | Mutable state in place | Each step = new snapshot with `model_copy` |
| Dependencies | langchain-core, pydantic, many more | Pydantic v2, typing_extensions only |
| API surface | Hundreds of classes | ~20 public exports |

GraphForge takes a **third path**: keep the graph model, enforce type safety from the ground up, make state management explicit, and minimise the surface area.

---

## Installation

```bash
pip install graphforge
```

**Optional dependencies:**

| Extra | Packages | Purpose |
|---|---|---|
| `[a2a]` | aiohttp | A2A protocol server |
| `[mcp]` | mcp | MCP integration |
| `[tracing]` | opentelemetry-api | OpenTelemetry tracing |
| `[store-redis]` | redis | Redis-backed store |

Requires Python 3.9+ and Pydantic v2.

---

## Core Concepts

### GraphState

State is the data flowing through your graph. Define it as a Pydantic v2 `BaseModel` and declare merge semantics per field:

```python
from graphforge import GraphState, node_field

class AgentState(GraphState):
    status: str = "idle"                                      # overwrite (default)
    messages: list = node_field(default=[], merge="append")   # append
    total: int = node_field(default=0, merge="reduce",        # custom reducer
                             reducer=lambda o, n: (o or 0) + n)
```

| Strategy | Behaviour | Typical Use |
|---|---|---|
| `overwrite` (default) | Old value replaced by new | Scalar fields, flags |
| `append` | New items appended to list | Message history, traces |
| `reduce` | Custom `(old, new) -> value` | Accumulators, counters |

**States are immutable** — use `state.apply(**updates)` to produce a new snapshot.

### Nodes

```python
def my_node(state: AgentState) -> dict:
    """Receives state, returns fields to update."""
    return {"messages": Append([new_msg]), "status": "done"}
```

Nodes can also be:
- **Async**: `async def node(state): ...`
- **Generators** (token streaming): `def node(state): yield update1; yield update2`
- **Subgraphs**: embed a compiled graph as a node
- **Pipelines**: embed a linear `Pipeline` as a node

### Edges

```python
graph.add_edge("node_a", "node_b")                     # unconditional
graph.add_error_edge("node_a", "fallback")              # error recovery
graph.add_fanout("node_a", ["b", "c"], join="d")        # parallel fan-out

graph.add_conditional_edges(                            # conditional routing
    "classify",
    router=lambda s: "process" if s.has_data else "wait",
    path_map={"process": "processor", "wait": "input_handler"},
)
```

### Graph & CompiledGraph

```python
graph = (
    Graph[AgentState]()
    .add_node("llm", llm_call)
    .add_node("tools", tool_executor)
    .add_edge("llm", "tools")
    .add_edge("tools", "__end__")
    .set_entry_point("llm")
    .compile(name="my_agent")
)
```

API shortcuts:

```python
graph.add_sequence(["a", "b", "c"])          # chain nodes in sequence
graph.add_parallel("start", ["a", "b"])      # parallel fan-out
```

### Pipeline

```python
from graphforge import Pipeline

pipe = Pipeline[AgentState]([step_one, step_two, step_three], name="preprocess")
graph.add_node("preprocess", pipe)
```

### Command

Nodes can dynamically route execution via `Command`:

```python
from graphforge import Command

def router_node(state):
    return Command(goto="search_tool", update={"query": state.input})
```

---

## Agent Features

### ReAct Agent

```python
from graphforge.agents import create_react_agent

def llm(messages, tools):
    return {"content": "Result", "tool_calls": []}

graph = create_react_agent(llm, tools=tools)
compiled = graph.compile(state_type=ReactState)
result = compiled.invoke(ReactState(messages=[{"role": "user", "content": "Hi"}]))
```

### ToolNode

```python
from graphforge.agents import ToolNode

tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
    "_func": lambda q: f"found: {q}",
}]

graph.add_node("agent", ToolNode(llm, tools=tools))
```

### @tool Decorator

Auto-generate tool schemas from Python type annotations:

```python
from graphforge.tools import tool

@tool
def search(query: str, max_results: int = 10) -> str:
    """Search the web."""
    return f"Results for {query}"

# search.tool_def -> OpenAI-compatible ToolDef
graph.add_node("agent", ToolNode(llm, tools=[search.tool_def]))
```

### Structured Output

Enforce LLM outputs conform to a Pydantic schema:

```python
from pydantic import BaseModel
from graphforge.structured_output import with_structured_output

class Weather(BaseModel):
    city: str
    temperature: float

llm = with_structured_output(my_llm_func, Weather, max_retries=3)
result = llm(messages)  # returns validated Weather instance
```

### Multi-Agent Patterns

```python
from graphforge.agents import create_supervisor_worker, create_swarm, create_delegation_agent

# Supervisor/Worker
graph = create_supervisor_worker(supervisor_fn, {"worker_a": worker_a_fn})

# Swarm (agent handoff)
graph = create_swarm({"agent_a": fn_a, "agent_b": fn_b}, router_fn)

# Delegation (sub-agents)
graph = create_delegation_agent(orchestrator_fn, {"sub": sub_graph})
```

### Approval Patterns

Wrap any node with a human approval gate:

```python
from graphforge.agents import ApprovalNode

graph.add_node("send_email", ApprovalNode(send_email_fn, timeout=3600))
```

---

## Streaming & Execution

### Execution Modes

```python
result = compiled.invoke(state)                                    # sync
result = await compiled.ainvoke(state)                             # async

for event in compiled.stream(state):                               # sync stream
    async for event in compiled.astream(state):                    # async stream
```

### Streaming Modes

```python
# "events" — StreamEvent objects (default, backward compatible)
# "values" — full state after each node
# "updates" — only the updates dict
# "debug"  — full metadata with timing

for event in compiled.stream(state, stream_mode="values"):
    print(event)
```

### Token-Level Streaming

Nodes can be generator functions for token-by-token output:

```python
def token_generator(state):
    for token in ["Hello", " ", "World"]:
        yield {"output": token}

graph.add_node("stream_gen", token_generator)

for event in compiled.stream(state):
    if event.type == EventType.STREAM_TOKEN:
        print(event.data["token"])
```

### Cancellation

```python
# From another thread:
compiled.cancel("thread-id")
compiled.clear_cancel("thread-id")
```

### Configuration System

Override state fields at invocation time:

```python
result = compiled.invoke(state, configurable={"model": "gpt-4", "temperature": 0.9})
```

---

## Persistence & Memory

### Store (Long-Term Memory)

Cross-thread, cross-session persistent memory — independent of checkpoint state:

```python
from graphforge import InMemoryStore, Store

store = InMemoryStore()
store.put("session-123", "prefs", {"theme": "dark"})
prefs = store.get("session-123", "prefs")

# Redis backend:
from graphforge.store_redis import RedisStore
store = RedisStore(host="localhost", port=6379)
```

Store is injectable into nodes:

```python
def my_node(state, store):
    prefs = store.get("user_1", "prefs")
    return {"output": prefs}

compiled.invoke(state, store=store)
```

### Checkpointing

```python
from graphforge import InMemoryCheckpointer, SqliteCheckpointer, RedisCheckpointer
from graphforge._checkpoint_postgres import PostgresCheckpointer

compiled = graph.compile(checkpointer=SqliteCheckpointer("checkpoints.db"))
compiled.invoke(state, config={"thread_id": "session-1"})
```

| Backend | Use Case |
|---|---|
| `InMemoryCheckpointer` | Development, testing |
| `SqliteCheckpointer` | Single-process production |
| `RedisCheckpointer` | Distributed deployments |
| `PostgresCheckpointer` | Production with existing PG |

### Node-Level Checkpoint Skipping

```python
# Skip checkpoint for pure functions (performance optimization)
graph.add_node("transform", clean_data, checkpoint=False)
graph.add_node("llm_call", call_llm, checkpoint=True)  # default
```

---

## Safety & Control

### Guardrails

```python
from graphforge.guardrails import InputGuardian, OutputGuardian, FieldLengthGuardrail

guardian = InputGuardian([
    FieldLengthGuardrail("prompt", max_length=5000),
])
guardian.check({"prompt": "user input..."})  # raises GuardrailError if blocked

# Custom guardrail:
class PIIFilter:
    def check_input(self, state):
        if "ssn:" in str(state):
            return GuardrailResult.block("PII detected")
        return GuardrailResult.allow()
```

### Human-in-the-Loop

```python
from graphforge import interrupt

def review_node(state):
    response = interrupt(
        message="Approve this action?",
        value={"action": "send_email"},
        timeout=3600,
        on_timeout="reject",
    )
    return {"approved": response.get("decision") == "approve"}

# Resume with:
# compiled.resume("thread-id", updates={"decision": "approve"})
```

### State Middleware

```python
from graphforge._middleware import MiddlewarePipeline

class Logger:
    def pre_update(self, node, state, updates):
        print(f"{node}: {list(updates.keys())}")
        return updates

pipeline = MiddlewarePipeline([Logger()])
```

### Cancellation

```python
threading.Thread(target=lambda: compiled.invoke(state, config={"thread_id": "x"})).start()
compiled.cancel("x")
```

---

## RAG (Retrieval-Augmented Generation)

Built-in module for knowledge retrieval with embeddings, vector stores, and retrieval nodes:

```python
from graphforge.rag import InMemoryVectorStore, RetrievalNode, chunk_text
from graphforge.rag._embeddings import DeterministicEmbeddings

embeddings = DeterministicEmbeddings(dimension=128)
store = InMemoryVectorStore(embeddings)

store.add_texts([
    "Paris is the capital of France",
    "Python is a programming language",
])

graph.add_node("retrieve", RetrievalNode(store, query_field="question"))
```

**Chunking:**

```python
from graphforge.rag import chunk_text, ChunkingStrategy

chunks = chunk_text(long_text, strategy="recursive", chunk_size=500)
store.add_texts(chunks)
```

---

## Developer Tools

### TimelineRecorder — Execution Debugger

Record every state transition for post-mortem debugging:

```python
from graphforge import CallbackManager
from graphforge._timeline import TimelineRecorder

recorder = TimelineRecorder()
compiled.invoke(state, callbacks=CallbackManager([recorder]))

for frame in recorder.get_timeline():
    print(f"{frame.node}: {frame.duration:.3f}s")

recorder.export_json("trace.json")
for frame in recorder.replay():
    print(frame.node, frame.state_after)
```

### AutoOptimizer — Automatic Parallelization

Static analysis that detects independent paths and suggests optimizations:

```python
from graphforge._optimizer import optimize, auto_parallelize

report = optimize(compiled_graph)
print(report.summary())
# Independent paths: 2
# Unused nodes: 0
# Bottlenecks: 1

# Apply automatically:
optimized = auto_parallelize(graph_builder, state_type)
```

### Evaluation

```python
from graphforge.eval import evaluate, EvalCase, exact_match, contains

cases = [
    EvalCase(
        input={"messages": [{"role": "user", "content": "Hi"}]},
        expected={"output": "Hello"},
        metrics=[exact_match("output")],
    ),
]
results = evaluate(compiled_graph, cases, state_type=ChatState)
print(results.summary())  # "EvalResults: 3/4 passed (75.0%)"
```

### Cost Tracking

```python
from graphforge._callbacks import CostCallback

cost = CostCallback()
compiled.invoke(state, callbacks=CallbackManager([cost]))

print(f"Total: ${cost.total_cost():.4f} ({cost.total_tokens()} tokens)")
for node, stats in cost.get_stats().items():
    print(f"  {node}: ${stats['cost']:.4f}")

cost.set_pricing("my-model", input_price=0.01, output_price=0.03)
cost.track("gpt-4", prompt_tokens=100, completion_tokens=50, node="llm")
```

### TimingCallback

```python
from graphforge._callbacks import TimingCallback

timer = TimingCallback()
compiled.invoke(state, callbacks=CallbackManager([timer]))

for node, stats in timer.get_stats().items():
    print(f"{node}: {stats['duration']:.3f}s ({stats['calls']} calls)")
```

### CLI

```bash
python -m graphforge.cli info graph.json      # Show topology
python -m graphforge.cli viz graph.json       # Export visualization
python -m graphforge.cli run graph.json state.json  # Run graph
```

---

## Deployment

### serve() — One-Command API Server

Turn any compiled graph into a production-ready API server with one function call:

```python
from graphforge import serve

serve(compiled_graph)                           # REST + WS + MCP + A2A
serve(graph, port=9090, api_key="sk-...")       # Custom port + auth
```

**Endpoints:**

| Endpoint | Protocol | Description |
|---|---|---|
| `POST /invoke` | REST | Synchronous invocation |
| `POST /stream` | SSE | Streaming events |
| `GET /ws` | WebSocket | Bidirectional streaming |
| `GET /health` | REST | Health check |

### A2A Protocol

Agent-to-Agent communication (Google protocol):

```python
from graphforge.a2a import A2AServer

server = A2AServer(compiled_graph, host="0.0.0.0", port=8081)
server.run()
```

### MCP Integration

Connect to any MCP-compatible server:

```python
from graphforge.mcp import MCPClient, MCPAgentServer

client = MCPClient("npx", args=["-y", "mcp-server"])
async with client:
    tools = await client.list_tools()
```

Or expose your graph as MCP tools:

```python
server = MCPAgentServer(compiled_graph)
server.serve()
```

### Background Execution

```python
from graphforge.background import BackgroundTaskRunner

runner = BackgroundTaskRunner(max_workers=4)
task = runner.submit(graph, state)
result = task.wait(timeout=30)
print(task.status)  # "completed" | "failed"
```

---

## Observability

### Callbacks

```python
from graphforge import Callback, CallbackManager

class Logger(Callback):
    def on_node_start(self, node, state): print(f"  starting {node}")
    def on_node_end(self, node, state):   print(f"  finished {node}")

compiled.invoke(state, callbacks=CallbackManager([Logger()]))
```

| Hook | Called When |
|---|---|
| `on_graph_start(name, input)` | Execution begins |
| `on_graph_end(name, final)` | Execution completes |
| `on_node_start(node, state)` | Just before node runs |
| `on_node_end(node, state)` | Just after node succeeds |
| `on_node_error(node, error)` | Node raises exception |
| `on_state_update(node, updates, new_state)` | State is merged |
| `on_conditional_edge(node, result, target)` | Router evaluated |

### Logging

```python
from graphforge import configure_logging
import logging

configure_logging(level=logging.DEBUG)
```

### OpenTelemetry Tracing

```python
from graphforge import TracingCallback

compiled.invoke(state, callbacks=CallbackManager([TracingCallback()]))
```

---

## Comparison with LangGraph

| Area | LangGraph | GraphForge |
|---|---|---|
| State model | TypedDict + magic `__reducers__` | Pydantic v2, explicit `node_field(merge=...)` |
| Type safety | Weak runtime TypedDict | Full generics, Protocols, static analysis |
| Immutability | Mutable in place | Each step = new snapshot (`model_copy`) |
| Graph composition | Channels-based | Subgraphs, Pipelines, add_sequence/parallel |
| Streaming | Opaque channel events | StreamEvent + values/updates/debug modes |
| Async | API exists | Native sync+async from day 1 |
| Checkpointing | BaseCheckpointSaver | InMemory / SQLite / Redis / Postgres |
| MCP | Native | Client + Server |
| Store | BaseStore | Store ABC + InMemory + Redis |
| Guardrails | Built-in | InputGuardian + OutputGuardian |
| Multi-Agent | Supervisor, Swarm | Supervisor, Swarm, Delegation |
| Tool decorator | @tool | @tool + auto JSON Schema |
| Structured output | with_structured_output | with_structured_output + retry |
| Cancellation | Task cancellation | cancel(thread_id) |
| Evaluation | LangSmith | Built-in EvalCase + evaluate() |
| WebSocket | Platform | Built-in |
| **serve() one-command** | ❌ | ✅ |
| **Auto parallelization** | ❌ | ✅ |
| **Timeline debugger** | ❌ only LangSmith | ✅ built-in |
| **Dependencies** | langchain-core + many | Pydantic v2 only |
| **Error handling** | Let propagate | Log + dispatch + retry + error edges |
| **Callbacks** | None | Callback Protocol + Manager |
| **State middleware** | None | MiddlewarePipeline |
| **Checkpoint skip** | Per-node | Per-node |
| **Configurable fields** | configurable_fields | invoke(configurable=...) |
| **Background tasks** | Platform | BackgroundTaskRunner |
| **CLI** | langgraph CLI | graphforge CLI |

---

## API Reference

### Graph[StateT]

| Method | Returns | Description |
|---|---|---|
| `add_node(name, fn, retry=0, timeout=None, metadata=None, checkpoint=True)` | Self | Register a node |
| `add_edge(source, target)` | Self | Unconditional edge |
| `add_error_edge(source, fallback)` | Self | Error recovery route |
| `add_conditional_edges(source, router, path_map)` | Self | Conditional routing |
| `add_fanout(source, targets, join=None, conflict=None)` | Self | Parallel fan-out |
| `add_sequence([a, b, c])` | Self | Linear chain |
| `add_parallel(source, [a, b], join=j)` | Self | Parallel fan-out |
| `set_entry_point(name)` | Self | Start node |
| `set_finish_point(name)` | Self | Terminal node |
| `set_metadata(key, value)` | Self | Attach metadata |
| `compile(...)` | CompiledGraph | Freeze and validate |

### CompiledGraph[StateT]

| Method | Returns | Description |
|---|---|---|
| `invoke(state, config=None, callbacks=None, store=None, configurable=None)` | StateT | Sync execution |
| `ainvoke(state, config=None, callbacks=None, store=None, configurable=None)` | StateT | Async execution |
| `stream(state, config=None, callbacks=None, store=None, stream_mode="events")` | Generator | Sync streaming |
| `astream(state, config=None, callbacks=None, store=None, stream_mode="events")` | AsyncGenerator | Async streaming |
| `resume(thread_id, state_type=None, updates=None, config=None, callbacks=None, store=None)` | StateT | Resume from checkpoint |
| `cancel(thread_id)` | None | Cancel running execution |
| `clear_cancel(thread_id)` | None | Clear cancellation |

### Symbols

```
Graph, CompiledGraph, Node, NodeKind,
GraphState, Append, MergeStrategy, node_field,
Pipeline,
EventType, StreamEvent, StreamMode,
Checkpoint, Checkpointer, CheckpointKey, InMemoryCheckpointer,
SqliteCheckpointer, RedisCheckpointer, PostgresCheckpointer,
GraphExecutionPaused,
Callback, CallbackManager, TimingCallback, CostCallback,
Store, InMemoryStore, RedisStore,
Guardrail, GuardrailAction, GuardrailError, GuardrailResult,
InputGuardian, OutputGuardian, FieldLengthGuardrail,
MapReduce,
Tool, tool, StructuredOutputWrapper, with_structured_output,
EvalCase, EvalResults, evaluate, exact_match, contains, json_match,
MCPClient, MCPAgentServer,
serve, UnifiedServer,
configure_logging, get_logger,
export_dot, render_graph,
NodeFunc, AsyncNodeFunc, RouterFunc, AsyncRouterFunc,
StreamingNodeFunc, AsyncStreamingNodeFunc,
NodeName, StateUpdate, ConfigDict
```

---

## Contributing

```bash
git clone https://github.com/carlos07281988/GraphForge.git
cd GraphForge
pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
