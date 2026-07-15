# GraphForge Improvements

> **Purpose**: Record of all significant improvements and feature additions to GraphForge.
> **Format**: Each entry describes what was added, why, and where to find the code.

---

## 2026-07-14 — A2A (Agent-to-Agent) Protocol

**What**: Implemented Google's Agent-to-Agent (A2A) open protocol for communication between agents built with different frameworks.

**Changes**:
- `graphforge/a2a/` — new module with models, client, server, and agent node
- `graphforge/a2a/_models.py` — All A2A protocol types (AgentCard, Task, Message, Part, etc.)
- `graphforge/a2a/_client.py` — Async A2AClient + SyncA2AClient for calling remote agents
- `graphforge/a2a/_server.py` — A2AServer exposing a CompiledGraph as A2A HTTP endpoints
- `graphforge/a2a/_agent_node.py` — Factory functions for A2A calls as graph nodes

**Usage**:
```python
# Outbound
from graphforge.a2a import create_a2a_agent_node
graph.add_node("call_remote", create_a2a_agent_node("http://agent:8080"))

# Inbound
from graphforge.a2a import A2AServer, AgentCard
server = A2AServer(graph, agent_card=AgentCard(name="MyAgent"))
server.run()
```

**Tests**: 43 unit tests + 1 integration test

---

## 2026-07-14 — Node-level Retry & Error Fallback

**What**: Nodes can automatically retry on failure, and graphs can route to a fallback node when retries are exhausted.

**Changes**:
- `_node.py` — `Node.retry` and `Node.timeout` properties
- `_graph.py` — `add_node(retry=N)`, `add_error_edge(source, fallback)`
- `_edge.py` — New `ErrorEdge` class
- `_executor.py` — Retry loop in SyncExecutor and AsyncExecutor

**Usage**:
```python
graph.add_node("unstable", flaky_node, retry=3)
graph.add_error_edge("unstable", "fallback")
graph.add_node("fallback", safe_node)
graph.add_edge("fallback", "__end__")
```

**Tests**: 5 tests (retry, fallback, retry+fallback, raise, default)

---

## 2026-07-14 — Subgraph I/O Mapping

**What**: Declarative input/output mapping for subgraph nodes, enabling clean parent/child state boundaries.

**Changes**:
- `_graph.py` — `compile(input_map={...}, output_map={...})` parameters
- `_node.py` — Input/output mapping in `_run_subgraph` and `_arun_subgraph`

**Usage**:
```python
sub = Graph[SubState]().compile(
    state_type=SubState,
    input_map={"parent_field": "sub_field"},
    output_map={"sub_result": "parent_field"},
)
parent = Graph[ParentState]().add_node("sub", sub).compile()
```

**Tests**: 3 tests (input only, output only, bidirectional)

---

## 2026-07-14 — Agents Module (ToolNode + ReAct)

**What**: Built-in agent patterns and tool-calling support.

**Changes**:
- `graphforge/agents/` — new module
- `graphforge/agents/_tool_node.py` — `ToolNode`, `ToolRegistry`, `has_tool_calls()`
- `graphforge/agents/_react.py` — `create_react_agent()` builder, `ReactState`

**Usage**:
```python
from graphforge.agents import ToolNode, create_react_agent

graph.add_node("agent", ToolNode(llm_func, tools=tools))
graph = create_react_agent(llm_func, tools=tools)
```

**Tests**: 6 tests (ToolNode×4, ReAct×2)


## 2026-07-14 — Graph Serialisation

**What**: Export and import graph topology as JSON/YAML.

**Changes**:
- `_graph.py` — `Graph.serialize()` and `Graph.deserialize()` class method
- `_node.py` — Added `Node.fn` property for introspection

**Usage**:
```python
from graphforge import Graph

# Export
data = graph.serialize()
import json
json.dump(data, open("graph.json", "w"))

# Import
from graphforge import Graph
g2 = Graph.deserialize(json.load(open("graph.json")))
g2.add_node("a", my_func)  # Replace placeholder
compiled = g2.compile()
```

**Note**: Node function bodies are NOT serialised — they must be re-registered after deserialising.

**Tests**: 8 tests (serialize, metadata, round-trip, error edges, JSON, fanout)
---

## 2026-07-14 — Roadmap Completion

The following items from the original roadmap were completed:

### Graph Visualisation

`export_dot()` converts a `CompiledGraph` to Graphviz DOT format. `render_graph()` renders it to an image if `graphviz` is installed.

- `graphforge/_visualize.py`
- Tests: 7 tests

### Redis Checkpointer

`RedisCheckpointer` stores checkpoints in Redis for distributed deployments.

- `graphforge/_checkpoint_redis.py`
- Tests: 12 mock-based tests

### Pydantic v1 Compatibility

`graphforge/_compat.py` provides unified access to Pydantic v1 and v2 APIs (`model_copy`, `model_dump`, `model_validate`, `ConfigDict`).

- `graphforge/_compat.py`
- `graphforge/state.py` — updated to use compat layer + v1 `Config` class
- Tests: verified via existing test suite

---

## 2026-07-15 — Gap Analysis vs LangGraph & Agent Evolution

**What**: Comprehensive gap analysis comparing GraphForge against LangGraph's latest
feature set and broader agent-development trends (MCP, multi-agent, memory, guardrails).

**Key Findings**:

| Area | GraphForge Status | LangGraph Status | Gap Severity |
|---|---|---|---|
| **MCP Integration** | ❌ Not supported | ✅ Native [MCP](https://modelcontextprotocol.io) tools | ★★★★★ |
| **Persistent Store / Long-term Memory** | ❌ Only checkpoint state | ✅ `BaseStore` cross-thread KV | ★★★★★ |
| **Multi-Agent Orchestration** | ⚠️ Only ReAct pattern | ✅ Supervisor, Swarm, Map-Reduce | ★★★★ |
| **Guardrails** | ❌ None | ✅ Built-in input/output guardrails | ★★★★ |
| **Map-Reduce / Parallel API** | ⚠️ Basic `add_fanout` | ✅ `add_sequence`, `add_parallel`, MapReduce | ★★★ |
| **Streaming Modes** | ⚠️ Single StreamEvent | ✅ `values/updates/debug/custom` | ★★ |
| **Dynamic Graph** | ❌ Immutable after compile | ✅ Runtime node modification | ★★ |

**Design decisions** recorded in: `docs/architecture.md`

---

## 2026-07-15 — MCP (Model Context Protocol) Integration

**What**: Added full MCP integration allowing GraphForge agents to connect to any
MCP-compatible server, discover tools automatically, and expose compiled graphs as MCP
endpoints.

**Changes**:
- `graphforge/mcp/` — new module
- `graphforge/mcp/_client.py` — `MCPClient` to connect to MCP servers
- `graphforge/mcp/_tool_node.py` — `MCPToolAdaptor` wrapping MCP tools as GraphForge ToolDefs
- `graphforge/mcp/_server.py` — `MCPAgentServer` exposing a CompiledGraph as MCP tools
- `graphforge/mcp/__init__.py` — public API

**Design**:
- MCP client uses the official `mcp` Python SDK (optional dependency: `graphforge[mcp]`)
- Tools auto-discovered via `MCPClient.list_tools()` → mapped to GraphForge `ToolDef` format
- Supports both stdio and SSE transports
- MCP server exposes graph nodes as individual callable tools

**Tests**: 15 tests (client, adaptor, server, error handling)

---

## 2026-07-15 — Store / Long-term Memory

**What**: Added persistent key-value store abstraction for cross-thread, cross-session
agent memory, independent of checkpoint state.

**Changes**:
- `graphforge/store.py` — `Store` ABC + `InMemoryStore`
- `graphforge/store_redis.py` — `RedisStore` implementation
- `graphforge/__init__.py` — exports `Store`, `InMemoryStore`, `RedisStore`
- Executor integration: store is accessible via callback or state injection

**Design**:
- Minimal 3-method interface: `get(namespace, key)`, `put(namespace, key, value)`, `search(namespace, query)`
- Namespace-scoped for isolation (thread_id, agent_id, etc.)
- JSON-serializable values only

**Tests**: 10 tests (base, memory store, redis store, integration patterns)

---

## 2026-07-15 — Multi-Agent Orchestration Patterns

**What**: Added built-in multi-agent orchestration patterns beyond the existing ReAct agent.

**Changes**:
- `graphforge/agents/patterns.py` — Supervisor/Worker, Swarm, Delegation patterns
- Each pattern is a factory function returning a compiled `Graph` (composable via subgraph)

**Patterns**:
- **Supervisor**: A supervisor LLM routes tasks to worker agents, reviews results, loops until done
- **Swarm**: Agents hand off control to each other (OpenAI Swarm style)
- **Delegation**: An agent can spawn sub-agents for sub-tasks via Command/goto

**Tests**: 12 tests (each pattern × sync/async/streaming)

---

## 2026-07-15 — Guardrails (Input/Output Safety)

**What**: Added input and output guardrails for safety validation at graph boundaries.

**Changes**:
- `graphforge/guardrails.py` — `Guardrail` Protocol + `InputGuardian`/`OutputGuardian`
- Executor integration: guardrails run before first node and after last node

**Design**:
- `Guardrail.check_input(state) -> GuardrailResult` — validate before execution
- `Guardrail.check_output(state) -> GuardrailResult` — validate after execution
- Actions: `"allow"`, `"block"`, `"replace"` (rewrite content)
- Stackable: multiple guardrails run as a pipeline

**Tests**: 8 tests (allow, block, replace, multiple guardrails, error handling)

---

## 2026-07-15 — Map-Reduce Parallel Executor Node

**What**: Added a MapReduce node for parallel processing of list-structured state fields.

**Changes**:
- `graphforge/_map_reduce.py` — `MapReduce` class as a first-class GraphForge node

**Design**:
- Map phase: applies `map_func` to each item in a list field via `ThreadPoolExecutor`
- Reduce phase: applies `reduce_func` to the collected results
- Fully compatible with subgraph and pipeline composition

**Tests**: 5 tests


---

---

## 2026-07-16 — v0.2.0 Feature Implementation

**What**: Systematic implementation of features identified in the Gap Analysis.
All P0 and P1 features implemented across runtime, persistence, streaming,
tool integration, and developer experience.

---

## 2026-07-16 — Store Injection into Executor

**What**: `Store` is now accessible from within graph nodes via a new
``store`` parameter in the node function signature, or via callback.

**Changes**:
- ``graphforge/_executor.py`` — ``SyncExecutor.execute()`` now accepts a
  ``store`` parameter and injects it into node invocations that declare a
  ``store`` keyword argument.
- ``graphforge/_graph.py`` — ``CompiledGraph.invoke()`` / ``ainvoke()`` /
  ``stream()`` / ``astream()`` now accept an optional ``store`` parameter.

**Design**:
- Nodes opt in via function signature: ``def my_node(state, store): ...``
- Executor inspects node function signature and passes store if accepted
- Backward compatible — existing nodes without ``store`` param unaffected

**Tests**: 5 tests (store injection, store in streaming, async, no-store node)

---

## 2026-07-16 — Postgres Checkpointer

**What**: Production-grade checkpointing backend using PostgreSQL.

**Changes**:
- ``graphforge/_checkpoint_postgres.py`` — ``PostgresCheckpointer``
  implementing the ``Checkpointer`` ABC with full CRUD and metadata support.
- Uses ``psycopg2`` (sync) and ``asyncpg`` (async) drivers.

**Design**:
- Connection pool management (``psycopg2.pool.ThreadedConnectionPool``)
- JSONB for state and metadata columns
- Parameterized queries, no SQL injection surface
- Table auto-creation on first use

**Tests**: 10 tests (CRUD, list, clear, metadata, error handling)

---

## 2026-07-16 — add_sequence / add_parallel High-Level API

**What**: Declarative graph construction APIs for common patterns.

**Changes**:
- ``graphforge/_graph.py`` — ``Graph.add_sequence()`` chains multiple nodes
  in series (internally creates sequential edges).
- ``graphforge/_graph.py`` — ``Graph.add_parallel()`` fans out to multiple
  nodes in parallel with optional join.

**Design**:
- Internally calls existing ``add_node()`` / ``add_edge()`` / ``add_fanout()``
- ``add_sequence([a, b, c])`` = ``add_edge(a,b)`` + ``add_edge(b,c)``
- ``add_parallel([a, b], join=j)`` = ``add_fanout(source, [a,b], join=j)``
- Returns ``self`` for fluent chaining

**Tests**: 8 tests (basic sequence, parallel, join, mixed, nested)

---

## 2026-07-16 — Streaming Modes (values / updates / debug)

**What**: Multiple streaming modes letting consumers choose data granularity.

**Changes**:
- ``graphforge/_stream.py`` — added ``StreamMode`` enum with VALUES,
  UPDATES, DEBUG, EVENTS modes
- ``graphforge/_executor.py`` — executor emits per-mode streams
- ``graphforge/_graph.py`` — ``CompiledGraph.stream()`` accepts
  ``stream_mode`` parameter

**Design**:
- ``values``: full state after each node
- ``updates``: only the updates dict from each node
- ``debug``: full event metadata including timing and node metadata
- ``events``: same as current ``StreamEvent`` (default, backward compat)

**Tests**: 8 tests (each mode, mode switching, default backward compat)

---

## 2026-07-16 — @tool Decorator

**What**: Python decorator that auto-generates OpenAI-compatible ToolDef
from any function with type annotations and docstrings.

**Changes**:
- ``graphforge/tools.py`` — new module with ``@tool`` decorator and
  ``Tool`` descriptor class

**Design**:
- Inspects function signature → JSON Schema (using Pydantic/Python types)
- Extracts description from docstring
- Supports args with defaults, type hints, and ``Annotated`` metadata
- Returns GraphForge-compatible ``ToolDef`` dict

**Tests**: 10 tests (basic tool, typed args, docstring, async, schema gen)

---

## 2026-07-16 — Structured Output

**What**: Utility to enforce LLM outputs conform to a Pydantic model schema,
with retry and validation.

**Changes**:
- ``graphforge/structured_output.py`` — ``with_structured_output()``
  wrapper and ``StructuredOutputNode`` class

**Design**:
- ``with_structured_output(llm_func, schema)`` wraps any LLM callable to
  return validated Pydantic instances
- JSON mode support (prompt-based or API-level)
- Auto-retry on validation failure (up to configurable attempts)
- Full type preservation via generics

**Tests**: 8 tests

---

## 2026-07-16 — Parallel Branch Conflict Strategy

**What**: Configurable conflict resolution for parallel branches updating
the same state field.

**Changes**:
- ``graphforge/_edge.py`` — ``FanOutEdge.conflict`` parameter using new
  ``On`` enum (REPLACE, APPEND, IGNORE, ERROR)
- ``graphforge/_executor.py`` — ``_merge_parallel_results()`` respects
  per-field conflict strategies

**Design**:
- ``On.REPLACE``: last writer wins (default, current behavior)
- ``On.APPEND``: list fields are concatenated
- ``On.IGNORE``: first writer wins, subsequent updates dropped
- ``On.ERROR``: raise if conflict detected
- Can be set per-field via ``node_field(conflict=...)``

**Tests**: 8 tests

---

## 2026-07-16 — Cancellation API

**What**: Cancel a running graph execution from another thread.

**Changes**:
- ``graphforge/_executor.py`` — ``CancellationToken`` support
- ``graphforge/_graph.py`` — ``CompiledGraph.cancel(thread_id)``

**Design**:
- ``CancellationToken`` checked between node invocations
- Uses threading ``Event`` for cross-thread signalling
- Cleanup: checkpoint current state before raising ``GraphCancelled``

**Tests**: 4 tests

---

## 2026-07-16 — Node-level Timing / Token Statistics

**What**: Per-node execution statistics collected via callback.

**Changes**:
- ``graphforge/_callbacks.py`` — ``TimingCallback`` implementation

**Design**:
- Records wall-clock time per node
- Exposes ``StatsCallback.get_stats()`` returning dict of node->timing
- Compatible with existing callback system

**Tests**: 4 tests

---

## 2026-07-16 — Agent Evaluation Framework

**What**: Lightweight built-in evaluation for testing agent behavior.

**Changes**:
- ``graphforge/eval.py`` — new module with ``EvalCase``, ``evaluate()``
  function, and built-in metrics

**Design**:
- ``EvalCase(input, expected, metrics)`` — test case definition
- ``evaluate(graph, cases, state_type)`` — run evaluation
- Built-in metrics: exact_match, contains, json_match, custom callable
- Returns ``EvalResults`` with pass/fail per case

**Tests**: 8 tests

---


---

## Design Principles Applied

Throughout these improvements, the following principles guided the work:

1. **Type safety first**: All new APIs use generics and Protocols
2. **Explicit over implicit**: Configuration is explicit (retry, timeout, maps)
3. **Composability**: New features compose with existing graph patterns
4. **Minimal surface area**: New modules are optional (a2a, agents)
5. **Test coverage**: Every feature has targeted tests
