# GraphForge Architecture

> **Version**: 0.1.0
> **Status**: Design Reference
> **Last Updated**: 2026-07-13

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Core Abstractions](#2-core-abstractions)
3. [Module Dependency Graph](#3-module-dependency-graph)
4. [State Management Deep-Dive](#4-state-management-deep-dive)
5. [Graph Execution Lifecycle](#5-graph-execution-lifecycle)
6. [Streaming Architecture](#6-streaming-architecture)
7. [Checkpointing Model](#7-checkpointing-model)
8. [A2A (Agent-to-Agent) Protocol](#8-a2a-agent-to-agent-protocol)
8. [A2A (Agent-to-Agent) Protocol](#8-a2a-agent-to-agent-protocol)
9. [Comparison: GraphForge vs LangGraph vs LangChain](#9-comparison)
10. [Design Decisions & Rationale](#10-design-decisions--rationale)
11. [Extension Points](#11-extension-points)
12. [Glossary](#12-glossary)
8. [Comparison: GraphForge vs LangGraph vs LangChain](#8-comparison)
9. [Design Decisions & Rationale](#9-design-decisions--rationale)
10. [Extension Points](#10-extension-points)
11. [Glossary](#11-glossary)

---

## 1. Design Philosophy

GraphForge is built on a set of explicit design constraints that differentiate it from both LangGraph and LangChain:

### 1.1 Typed from the Ground Up

Every public API surface carries full generic type parameters. The `StateT` TypeVar flows through nodes, edges, graphs, executors, and checkpointers, enabling end-to-end type checking without casting. This eliminates an entire class of runtime errors that plague frameworks relying on duck-typed dicts.

### 1.2 Explicit State Contracts

State schemas are Pydantic v2 `BaseModel` subclasses. Each field declares its merge strategy explicitly via `node_field(merge=...)`. There are no implicit reducers, no `__reducers__` class-level magic, and no runtime introspection of TypedDict annotations. The contract between a node and the framework is: *"I return a dict of fields I want to change; you decide how to fold them in."*

### 1.3 Immutable Snapshots

Every invocation step produces a new state instance via `model_copy(update=..., deep=True)`. The framework never mutates state in place. This provides:

- **Determinism**: replaying a graph with the same inputs and checkpoints always produces the same execution path.
- **Debuggability**: every state snapshot is preserved as a checkpoint; you can inspect the state at any step.
- **Safety**: nodes cannot accidentally leak mutations across steps.

### 1.4 Minimal Surface Area

The framework exports exactly 7 public abstractions (Graph, CompiledGraph, GraphState, Node, Pipeline, StreamEvent, Checkpointer) plus support types. There is no `BaseLLM`, no `Chain`, no `Toolkit`, no `Memory`. These belong in user code or companion libraries, not the core runtime.

### 1.5 Composability over Inheritance

Graphs compose via subgraph embedding (a `CompiledGraph` can be used as a node in another graph) and pipeline embedding (a `Pipeline` can be a node). There is no class hierarchy to inherit from — just callables that consume state and return updates.

### 1.6 Transparent Execution

All execution paths (sync, async, streaming) emit structured events that can be consumed via callbacks or stream iterators. There is no hidden channel protocol or opaque internal queue.

---

## 2. Core Abstractions

### 2.1 GraphState

```
GraphState (Pydantic BaseModel)
  ├── apply(**updates) -> Self     # Immutable merge entry point
  ├── model_dump() -> dict          # Pydantic serialisation
  └── model_copy(update=...)        # Snapshot mechanism
```

Each field can carry a `ReducerDescriptor` attached via `json_schema_extra`, configuring how updates are folded:

| Strategy | Behaviour | Use Case |
|---|---|---|
| `overwrite` (default) | Replace old value with new | Scalar fields (names, counts, flags) |
| `append` | Extend existing list | Message history, path tracing |
| `reduce` | Call `(old, new) -> value` | Accumulators, custom merge logic |

### 2.2 Node

```
Node[StateT]
  ├── name: str
  ├── kind: NodeKind (FUNCTION | ASYNC | STREAM | SUBGRAPH | PIPELINE)
  ├── invoke(state) -> dict         # Sync execution
  ├── ainvoke(state) -> dict        # Async execution
  └── stream(state) -> Generator    # Streaming execution
```

A `Node` wraps any callable and classifies it at construction time. The classification (performed by `_classify()`) inspects the callable using `inspect` to determine whether it's sync, async, a generator, a `CompiledGraph`, or a `Pipeline`. This enables the executor to dispatch correctly without runtime type checks on every invocation.

### 2.3 Graph

```
Graph[StateT] (mutable builder)
  ├── add_node(name, fn)            # Register a node
  ├── add_edge(source, target)      # Unconditional edge
  ├── add_conditional_edges(src, router, path_map)  # Conditional routing
  ├── set_entry_point(name)         # Define start node
  ├── set_finish_point(name)        # Define terminal node
  └── compile(...) -> CompiledGraph # Freeze and validate
```

The `Graph` is deliberately mutable (builder pattern) while `CompiledGraph` is deliberately immutable. This split:

1. Allows incremental construction in any order.
2. Enables validation at compile time (dangling edges, missing entry point, invalid router targets).
3. Guarantees that a compiled graph's topology never changes during execution.

### 2.4 CompiledGraph

```
CompiledGraph[StateT] (immutable)
  ├── invoke(state) -> state        # Sync execution
  ├── ainvoke(state) -> state       # Async execution
  ├── stream(state) -> StreamEvent  # Streaming execution
  ├── astream(state) -> StreamEvent # Async streaming
  ├── nodes, entry_point, finish_points, checkpointer
  └── successors(name) -> [NodeName]
```

The compiled graph pre-computes successor tables and conditional edge lookup maps at construction time, enabling O(1) dispatch during execution.

### 2.5 Pipeline

```
Pipeline[StateT]
  ├── run(state) -> dict            # Sync sequential execution
  ├── arun(state) -> dict           # Async sequential execution
  └── steps: [Callable]
```

A `Pipeline` is a linear sequence of callables. Unlike a `Graph`, it has no branching, no routing, and no cycles. Every step receives the *accumulated* output of all previous steps. Pipelines can be embedded as nodes in graphs via `Graph.add_node(name, pipeline)`.

### 2.6 Edge Types

```
DirectEdge[StateT]                 ConditionalEdge[StateT]
  ├── source: str                    ├── source: str
  └── target: str                    ├── router: (state) -> str
                                     └── path_map: {str: str}
```

Edges are value objects — they carry no behaviour. The executor reads them during `_resolve_next()`.

### 2.7 Checkpointer

```
Checkpointer (ABC)
  ├── put(key, state, parent_key)
  ├── get(key) -> Checkpoint
  └── list(thread_id) -> [key]
```

The `Checkpointer` abstraction defines 3 operations. The framework ships with `InMemoryCheckpointer` for development; production deployments can implement backends backed by S3, Redis, SQLite, or Postgres.

### 2.8 Callback

```
Callback (Protocol)
  ├── on_graph_start/end
  ├── on_graph_error
  ├── on_node_start/end/error
  ├── on_state_update
  └── on_conditional_edge
```

Every method is optional — the `CallbackManager` uses `hasattr` to check before dispatching. This avoids forcing users to implement empty method stubs.

---

## 3. Module Dependency Graph

```
                    ┌─────────────┐
                    │  __init__.py │  Public API surface
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┬────────────┬────────────┬───────────┐
              v            v            v            v            v           v
        ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐
        │ _types   │ │ _logging│ │ pipeline │ │  store   │ │ guardrails│ │ mcp/   │
        └──────────┘ └─────────┘ └────┬─────┘ └──────────┘ └──────────┘ └─────────┘
              │                       │
              v                       v
        ┌──────────┐             ┌──────────┐
        │  state   │             │  _node   │
        └──────────┘             └────┬─────┘
              │                       │
              v                       v
        ┌──────────┐             ┌──────────┐
        │ _edge    │◄────────────│  _graph  │
        └──────────┘             └────┬─────┘
              │                       │
              v                       v
        ┌──────────┐             ┌──────────┐
        │_checkpoint│            │_executor │
        └──────────┘             └──────────┘
              │                       │
              v                       v
        ┌──────────┐             ┌──────────┐
        │_checkpoint│            │_map_reduce│
        │_sqlite   │            └──────────┘
        │_redis    │
        └──────────┘
              │
              v
        ┌──────────┐
        │ store    │
        │ _redis   │
        └──────────┘
              │
              v
        ┌──────────┐
        │ agents/  │
        │  _react  │
        │  _tool   │
        │  patterns│
        └──────────┘
```python
# In merge_state:
if isinstance(new_val, Append):
    resolved[key] = old_val + list(new_val)  # Extend
else:
    resolved[key] = [*old_val, *new_val]     # Concatenate
```

This allows nodes to return plain lists for batch extension or `Append([single_item])` for incremental addition, both triggering the same merge behaviour. The marker is *not* required for append fields — any iterable is handled — but using `Append` makes the intent explicit.

---

## 5. Graph Execution Lifecycle

### 5.1 Synchronous Execution (`SyncExecutor.execute()`)

```
invoke(state)
  │
  ├── build config (recursion_limit, thread_id)
  ├── on_graph_start()
  │
  ├── [loop] while node != __end__:
  │     ├── check recursion_limit
  │     ├── get_node(name)
  │     ├── on_node_start()
  │     ├── node.invoke(state)  ────►  returns StateUpdate
  │     │     └── [error] on_node_error() + raise
  │     ├── _apply(state, updates)  ──►  new_state
  │     ├── on_state_update()
  │     ├── on_node_end()
  │     ├── [checkpointer] put(key, new_state)
  │     ├── state = new_state
  │     ├── _resolve_next(graph, state)  ──►  next node name
  │     │     ├── [conditional] router(state) ──►  path_map[key]
  │     │     └── [direct] successors[0] or __end__
  │     └── [loop]
  │
  └── on_graph_end()
  return final_state
```

### 5.2 Resolution Order

`_resolve_next()` prioritises conditional edges over direct edges:

1. If the current node has a conditional edge, call the router on the current state and look up the result in `path_map`.
2. If the current node has direct successors, return the first (and only — multiple direct successors without a conditional edge is an error).
3. If the current node has no outgoing edges, return `__end__` (implicit termination).

### 5.3 Error Propagation

Node errors are:

1. Logged at ERROR level with full traceback via `logger.exception()`.
2. Dispatched to `Callback.on_node_error()`.
3. Re-raised to the caller.

There is no built-in retry mechanism — that is the responsibility of the caller or a wrapper node.

---

## 6. Streaming Architecture

### 6.1 Event Model

```
StreamEvent
  ├── type: EventType
  ├── node: str        -- node name (empty for graph-level events)
  ├── data: dict       -- payload (state snapshots, updates, errors)
  ├── step: int        -- sequential step counter
  ├── parent: str|None  -- subgraph parent (future)
  └── metadata: dict
```

### 6.2 Event Sequence for a Typical Graph

```
GRAPH_START  {graph: "my_graph"}
  NODE_START   {node: "a", state: {…}}       step 0
    CONDITIONAL  {node: "a", next: "b"}       step 0
  NODE_END     {node: "a", state: {…}}       step 0
  NODE_START   {node: "b", state: {…}}       step 1
  STATE_UPDATE {node: "b", updates: {…}}     step 1
  NODE_END     {node: "b", state: {…}}       step 1
GRAPH_END    {state: {…}}
```

### 6.3 Streaming Modes

- **`stream()`**: blocks on each node invocation, yields events synchronously.
- **`astream()`**: same contract but uses async generators, yielding control to the event loop between each node.

Both modes produce identical event structures. Users are free to consume events via `for event in graph.stream(state)` or `async for event in graph.astream(state)`.

### 6.4 Stream vs Callbacks

Streaming and callbacks are complementary:

- **Streaming** (pull model): caller controls pacing, useful for UI updates, live dashboards.
- **Callbacks** (push model): called immediately, useful for logging, metrics, telemetry.
- Both are emitted from the same points in the executor; they observe the same events.

---

## 7. Checkpointing Model

### 7.1 Key Structure

```
CheckpointKey = Tuple[thread_id: str, node_name: str, step_number: int]
```

Checkpoints are uniquely identified by (thread, node, step), forming an append-only log. The `parent_key` field links checkpoints into a tree structure, enabling fork/join patterns (future).

### 7.2 Checkpoint Lifecycle

1. Before each node invocation, the current state is saved as a checkpoint keyed by the *current* (about-to-execute) node name.
2. The checkpoint includes the full state snapshot (via `model_dump()`), so the graph can be resumed from that exact point.
3. After the node executes, a new checkpoint is created with the updated state and linked to the previous via `parent_key`.

### 7.3 Resumption (Future)

The checkpoint model supports resumption:

```python
# Pseudocode for future resume()
checkpoints = checkpointer.list(thread_id)
last = checkpoints[-1]
snapshot = checkpointer.get(last)
graph.resume(snapshot.state, from_node=last[1])
```

This enables long-running agents and human-in-the-loop workflows.

---


## 8. A2A (Agent-to-Agent) Protocol

GraphForge implements Google's [Agent-to-Agent (A2A) open protocol](https://google.github.io/A2A/)
as a first-class module at `graphforge/a2a/`. This enables GraphForge agents to communicate
with agents built on any other framework that also supports the A2A standard.

### 8.1 Module Layout

```
graphforge/a2a/
├── __init__.py          # Public API exports
├── _models.py           # Pydantic v2 models: AgentCard, Task, Message, Part, etc.
├── _client.py           # A2AClient (async) + SyncA2AClient
├── _server.py           # A2AServer — HTTP server wrapping a CompiledGraph
└── _agent_node.py       # Node factories for outbound A2A calls
```

### 8.2 Outbound Integration

The `A2AClient` connects to any remote A2A agent via the standard task endpoints.
The `create_a2a_agent_node()` factory wraps this client as a graph node, so calling
a remote agent looks like any other node in your graph.

Data flow:
```
Graph State  ──→  input_mapper  ──→  A2A Message  ──→  HTTP POST /tasks/send
                                                  ┌──  A2A Task (response)
State Update  ←──  output_mapper  ←──  Message     ←──┘
```

### 8.3 Inbound Integration

The `A2AServer` wraps a `CompiledGraph` and exposes its A2A-compatible endpoints
on an HTTP port. Other agents (regardless of framework) can discover, invoke,
and monitor the graph via the standard protocol.

Data flow:
```
A2A Client  ──→  POST /tasks/send  ──→  state_factory  ──→  GraphState
                                                              │
                                                          CompiledGraph
                                                              │
A2A Client  ←──  Task (response)  ←──  result_mapper  ←────  GraphState (final)
```

### 8.4 Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/.well-known/agent-card` | Agent discovery document |
| POST | `/tasks/send` | Create task and wait for completion |
| POST | `/tasks/sendStream` | Create task with SSE streaming |
| GET | `/tasks/{id}` | Query task status |
| POST | `/tasks/{id}/cancel` | Cancel a running task |

### 8.5 Protocol Types

All A2A protocol types are defined as Pydantic v2 models in `_models.py`:

- **Part**: `TextPart`, `DataPart`, `FilePart` — message content units
- **Message**: role (user/agent) + list of parts
- **Task**: ID + status (submitted/working/input-required/completed/failed/canceled)
- **AgentCard**: name, description, capabilities, authentication, endpoints
- **TaskSendRequest/Response**: Standard request/response envelope

### 8.6 Authentication

The `A2AServer` supports Bearer token authentication via the `api_key` parameter.
Clients must include `Authorization: Bearer <api_key>` in their requests.

The `A2AClient` accepts an `api_key` parameter that is sent on every request.

### 8.7 Dependencies

The A2A module is optional. Install with:

```bash
pip install graphforge[a2a]
```

This adds `aiohttp>=3.9` as a runtime dependency.

## 9. Comparison

| Aspect | LangGraph | LangChain | GraphForge |
|---|---|---|---|
| **State model** | TypedDict + `__reducers__` magic | BaseMessage / memory | Pydantic v2 + `node_field(merge=...)` |
| **Type safety** | Weak — TypedDict has no runtime validation | Medium — some generics | Strong — `Generic[StateT]` end-to-end |
| **Immutability** | Mutable state in-place | Mutable | Immutable snapshots |
| **Merge strategy** | Global reducers per key | Implicit | Per-field explicit (overwrite/append/reduce) |
| **Graph composition** | Channels-based internals | N/A (no graph) | Subgraphs + Pipelines as first-class nodes |
| **Streaming** | Opaque channel events | Runnable events | Per-step `StreamEvent` objects |
| **Async support** | Async API exists | Async API | Native sync+async from day 1 |
| **Checkpointing** | `BaseCheckpointSaver` | N/A | `Checkpointer` ABC + `InMemoryCheckpointer` |
| **Callbacks** | None | `BaseCallbackHandler` hierarchy | Single `Callback` Protocol |
| **Exception handling** | Let-unhandled propagate | Try/except wrappers | Log + dispatch to callbacks + re-raise |
| **Dependencies** | langchain-core, pydantic | Many | Pydantic v2, typing_extensions |

### Key Design Wins

1. **No TypedDict magic**: LangGraph's `__reducers__` requires runtime annotation parsing that fails silently with incorrect types. GraphForge uses Pydantic's well-defined field metadata.
2. **No multi-class callback hierarchy**: LangChain has 15+ callback classes (BaseCallbackHandler, AsyncCallbackHandler, etc.). GraphForge has one Protocol with optional methods.
3. **No hidden state channels**: LangGraph's channel-based internals are complex and opaque. GraphForge's state is just a dict merged per explicit rules.
4. **No BaseMessage coupling**: GraphForge is message-format-agnostic. Your state schema defines exactly the fields you need.

### Limitations (Known)

1. **No built-in parallel execution** — nodes run sequentially. Parallel dispatch is a future goal.
2. **No visual debugging** — no graphviz/networkx export yet.
3. **No retry logic** — errors propagate immediately.
4. **In-memory checkpointer only** — production backends need custom implementations.
5. **Python 3.9 minimum** — some modern typing features (PEP 695, `Self` in `typing`) require `typing_extensions`.

---

## 10. Design Decisions & Rationale

### 9.1 Why Pydantic v2 vs TypedDict?

**Decision**: State schema = Pydantic `BaseModel`.

**Rationale**: 
- Pydantic provides built-in validation, serialisation (`model_dump()`), and schema generation.
- TypedDict is a *static* annotation with no runtime behaviour — LangGraph has to implement its own reducer mechanism using class-level `__reducers__` dicts and runtime annotation parsing, which is fragile.
- `model_copy(update=..., deep=True)` gives us immutable snapshots for free.
- The `json_schema_extra` field on Pydantic fields is a clean way to attach merge metadata without mucking with metaclasses.

### 9.2 Why Separate Graph and CompiledGraph?

**Decision**: Two classes instead of one with a `compile()` flag.

**Rationale**:
- The builder (`Graph`) is mutable — you add nodes and edges in any order.
- The compiled graph (`CompiledGraph`) is immutable — once frozen, the topology never changes.
- This separation allows the compiled graph to pre-compute lookup tables (successors, conditionals) at compile time, not at execute time.
- It mirrors the mental model of compilers: "build-time" vs "run-time".

### 9.3 Why Caching Reducer Map as ClassVar?

**Decision**: `_reducers: ClassVar[Optional[_ReducerMap]] = None`, lazily populated.

**Rationale**:
- Building the reducer map requires iterating over `model_fields` and checking `json_schema_extra` — cheap but not free.
- Doing it once per class (not per instance) is safe because the field metadata is class-level.
- Lazy population avoids eager introspection at module import time.
- The ClassVar paradigm also works correctly with subclasses (each subclass gets its own `_reducers` if it redefines `apply`, though in practice they share through `type(self).apply`).

### 9.4 Why `from __future__ import annotations`?

**Decision**: All modules use PEP 563 postponed evaluation of annotations.

**Rationale**:
- Enables forward references without string quotes in most places (e.g., `def invoke(self, state: StateT) -> StateUpdate:` without quoting `StateT` when it's defined later).
- Avoids circular import issues at module level — annotations are evaluated lazily.
- Consistent with modern Python best practices (PEP 649 will eventually replace this, but for 3.9-3.12, PEP 563 is the standard).

### 9.5 Why hasattr in CallbackManager?

**Decision**: `CallbackManager` checks `hasattr(cb, "method_name")` instead of requiring subclasses.

**Rationale**:
- The `Callback` is a Protocol, not an ABC. Users can implement just the methods they need.
- `hasattr` is cheap and lets us avoid maintaining a separate registry of "available methods".
- The alternative — requiring all callbacks to subclass a base — forces empty method stubs, which is noisy.

### 9.6 Why InMemoryCheckpointer as Default?

**Decision**: `graph.checkpointer or InMemoryCheckpointer()` in the executor.

**Rationale**:
- Checkpointing should be opt-out, not opt-in. Even in-memory checkpointing provides debuggability and the ability to inspect state history.
- `InMemoryCheckpointer` adds almost zero overhead (it's a dict).
- Users who don't want checkpointing can swap to a no-op checkpointer.

---

## 11. Extension Points

### 10.1 Custom State Field Types

Add new merge strategies by subclassing `ReducerDescriptor` and handling the new strategy in `merge_state()`:

```python
class AdditiveSet(ReducerDescriptor):
    """Union-merge for set fields."""
    strategy = MergeStrategy.REDUCE
    def __init__(self):
        super().__init__(strategy=MergeStrategy.REDUCE, func=lambda old, new: (old or set()) | (new or set()))
```

### 10.2 Custom Checkpointers

Implement the `Checkpointer` ABC:

```python
class RedisCheckpointer(Checkpointer[StateT]):
    def put(self, key, state, parent_key=None):
        redis.set(json.dumps(key), json.dumps(state))
    def get(self, key):
        data = redis.get(json.dumps(key))
        return Checkpoint(key=key, state=json.loads(data)) if data else None
    def list(self, thread_id):
        return [json.loads(k) for k in redis.scan_iter(f"*{thread_id}*")]
```

### 10.3 Custom Logging

Use Python's standard `logging` configuration:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Or:
from graphforge import configure_logging
configure_logging(level=logging.DEBUG)
```

---



## 12. New Modules (v0.2.0)

### 12.1 MCP (Model Context Protocol)

The ``mcp/`` module enables GraphForge to connect to any MCP-compatible server
and to expose compiled graphs as MCP endpoints. This bridges GraphForge with
the broader MCP tool ecosystem (thousands of tools available via MCP servers).

Design decisions:
- Client and server are separate classes (``MCPClient``, ``MCPAgentServer``)
- MCP tools map to GraphForge ``ToolDef`` format for compatibility with ``ToolNode``
- Both stdio and SSE transports supported
- Optional dependency: ``graphforge[mcp]``

### 12.2 Store / Long-Term Memory

The ``store.py`` and ``store_redis.py`` modules provide a minimal 3-method
interface for persistent cross-session memory, separate from checkpoint state.

Design decisions:
- Namespace-scoped for isolation (thread_id, agent_id, user_id)
- JSON-serializable values only (for portability)
- ``Store`` ABC, ``InMemoryStore`` (dev), ``RedisStore`` (prod)

### 12.3 Multi-Agent Orchestration Patterns

The ``agents/patterns.py`` module provides factory functions for common
multi-agent coordination patterns beyond single-agent ReAct.

Design decisions:
- Each pattern is a factory that returns an un-compiled ``Graph``
- Patterns compose (can be used as subgraphs)
- Custom state classes per pattern (or user-supplied)

### 12.4 Guardrails

The ``guardrails.py`` module provides input/output validation at graph boundaries.

Design decisions:
- Minimal ``Guardrail`` Protocol (one or both of ``check_input``/``check_output``)
- Three actions: ALLOW, BLOCK, REPLACE
- Stackable: multiple guardrails run as a pipeline
- ``InputGuardian`` / ``OutputGuardian`` containers

### 12.5 MapReduce
### 12.6 Tool Decorator (tools.py)

The ``@tool`` decorator auto-generates OpenAI-compatible ToolDef schemas from
Python function type annotations and docstrings. Eliminates the need to write
JSON Schema by hand when defining tools.

### 12.7 Structured Output (structured_output.py)

Wraps any LLM callable to return validated Pydantic model instances, with
automatic retry on validation failure. Uses JSON mode prompting and extracts
structured data from markdown code blocks.

### 12.8 Agent Evaluation (eval.py)

Built-in evaluation framework for testing compiled graphs against expected
outcomes. Provides ``EvalCase``, ``evaluate()``, and built-in metrics
(``exact_match``, ``contains``, ``json_match``).

### 12.9 Cancellation API

Cross-thread graph cancellation via ``CompiledGraph.cancel(thread_id)``.
Uses ``threading.Event`` for signalling. Saves a checkpoint before raising.

### 12.10 TimingCallback

Collects per-node execution timing via the existing callback system.
Provides ``TimingCallback.get_stats()`` for node-level duration/calls data.

### 12.11 add_sequence / add_parallel

High-level graph construction APIs added to the ``Graph`` builder:
- ``add_sequence([a, b, c])`` — linear chain (calls ``add_edge`` between pairs)
- ``add_parallel(source, [a, b], join=j)`` — parallel fan-out (wraps ``add_fanout``)

### 12.12 Streaming Modes

The ``stream()`` method now accepts a ``stream_mode`` parameter:
- ``"events"`` (default) — current StreamEvent format (backward compatible)
- ``"values"`` — full state after each node
- ``"updates"`` — only the updates dict
- ``"debug"`` — full event metadata with timing

### 12.13 Postgres Checkpointer

Production-grade checkpointing via PostgreSQL (``_checkpoint_postgres.py``).
Uses ``psycopg2`` connection pool, JSONB columns, and parameterized queries.


The ``_map_reduce.py`` module provides parallel list processing as a first-class
graph node.

Design decisions:
- Map phase via ``ThreadPoolExecutor`` (parallel, configurable workers)
- Reduce phase synchronous
- Compatible with state dicts and Pydantic models


## 13. Glossary

| Term | Definition |
|---|---|
| **State** | A Pydantic v2 BaseModel subclass representing the accumulated data flowing through a graph. |
| **StateUpdate** | A `dict[str, Any]` returned by a node, containing fields to merge into state. |
| **Node** | A callable `(state) -> StateUpdate` wrapped with metadata (name, kind). |
| **Edge** | A connection between nodes — either unconditional (`DirectEdge`) or state-dependent (`ConditionalEdge`). |
| **Router** | A callable `(state) -> str` used by `ConditionalEdge` to pick the next node. |
| **Graph** | A mutable builder for directed execution graphs. |
| **CompiledGraph** | An immutable, compiled graph ready for execution. |
| **Pipeline** | A linear sequence of steps (no branching, no routing). |
| **Checkpoint** | A snapshot of state at a specific `(thread, node, step)`. |
| **Checkpointer** | An abstract interface for storing/retrieving checkpoints. |
| **Callback** | A Protocol for lifecycle hooks. |
| **StreamEvent** | A structured event emitted during graph execution. |
| **MergeStrategy** | How a field's updates are folded: `overwrite`, `append`, or `reduce`. |
| **ReducerDescriptor** | Metadata attached to a Pydantic field specifying its merge strategy. |
| **Append** | A list subclass marker that signals append-merge semantics. |
| **Executor** | The runtime component that walks the graph topology and invokes nodes. |
