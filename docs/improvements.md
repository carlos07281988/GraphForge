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

## 2026-07-16 — v0.3.0 Feature Implementation

**What**: Final round of feature implementation covering remaining gaps from
the Gap Analysis: token-level streaming, WebSocket, checkpoint skipping,
configuration system, managed values, background execution, CLI, state middleware.

---

## 2026-07-16 — Token-Level Streaming (Generator Nodes)

**What**: Executor now properly supports generator nodes — nodes declared as
generator functions that ``yield`` intermediate state updates (e.g., LLM
token-by-token output).

**Changes**:
- ``graphforge/_executor.py`` — ``SyncExecutor.stream()`` and ``AsyncExecutor.stream()``
  now detect generator nodes and call ``node.stream()`` / ``node.astream()``
  instead of ``node.invoke()``.
- ``graphforge/_stream.py`` — Added ``STREAM_TOKEN`` event type for individual
  token emissions.
- ``graphforge/_node.py`` — Ensure ``Node.stream()`` / ``astream()`` are
  properly dispatched by the executor.

**Design**:
- Executor inspects ``node.kind``: when ``STREAM`` or ``ASYNC_STREAM``,
  it calls the streaming method and yields each intermediate update
- Users write ``def my_node(state): yield update1; yield update2``
- The ``STREAM_TOKEN`` event allows fine-grained UI updates (typewriter effect)

**Tests**: 8 tests (sync generator, async generator, token events, mixed)

---

## 2026-07-16 — WebSocket Streaming Endpoint

**What**: GraphServer now exposes a WebSocket endpoint for bidirectional
real-time communication with graph executions.

**Changes**:
- ``graphforge/_http_server.py`` — Added ``/ws`` WebSocket endpoint.
- WebSocket messages follow the same ``StreamEvent`` format as SSE.

**Design**:
- Uses ``aiohttp.web.WebSocketResponse``
- Client sends JSON ``{"state": ..., "config": ...}`` to start execution
- Server streams ``StreamEvent`` objects as JSON lines
- Supports client disconnect handling

**Tests**: 3 tests (connect, stream, disconnect)

---

## 2026-07-16 — Node-Level Checkpoint Skipping

**What**: Nodes can now opt out of checkpointing by setting
``checkpointer=False``, avoiding unnecessary I/O for pure functions.

**Changes**:
- ``graphforge/_node.py`` — Added ``checkpoint`` property to ``Node``.
- ``graphforge/_graph.py`` — ``Graph.add_node()`` accepts
  ``checkpoint=True`` parameter.
- ``graphforge/_executor.py`` — Executor skips ``checkpointer.put()``
  for nodes with ``checkpoint=False``.

**Design**:
- Default ``True`` (backward compatible)
- Pure transform nodes (formatting, validation) can set ``checkpoint=False``

**Tests**: 4 tests (skip, default true, streaming with skip)

---

## 2026-07-16 — Configuration System (Configurable Fields)

**What**: State fields can now be marked as ``configurable``, allowing the
same compiled graph to be reused with different configuration values
without recompiling.

**Changes**:
- ``graphforge/state.py`` — ``node_field()`` now accepts ``configurable``
  parameter.
- ``graphforge/_graph.py`` — ``CompiledGraph.invoke()`` accepts a
  ``configurable`` dict that overrides configurable field values.

**Design**:
- Fields marked ``configurable=True`` can be overridden via
  ``invoke(state, configurable={"model": "gpt-4"})``
- Configurable values are merged into state before the first node runs
- Backward compatible — existing fields default ``configurable=False``

**Tests**: 5 tests

---

## 2026-07-16 — Managed Values (Parallel Branch Shared State)

**What**: Added managed values — shared state between parallel branches
that is automatically merged when branches converge.

**Changes**:
- ``graphforge/_edge.py`` — ``FanOutEdge.managed_values`` property
- ``graphforge/_executor.py`` — Executor tracks managed values across
  parallel branches and merges them at join points.

**Design**:
- Managed values are declared per fan-out edge
- Each branch can read and write managed values
- At join, managed values are merged using the field's merge strategy
- Separate from regular state — only appears in parallel contexts

**Tests**: 4 tests

---

## 2026-07-16 — Background Execution

**What**: Simple background task execution system for running graphs
in separate threads with status tracking.

**Changes**:
- ``graphforge/background.py`` — new module with ``BackgroundTaskRunner``
  and ``BackgroundTask`` classes.

**Design**:
- ``BackgroundTaskRunner.submit(graph, state) -> BackgroundTask``
- Tasks run in a thread pool
- Status tracking: pending/running/done/failed
- ``task.result()`` blocks until complete
- ``task.cancel()`` uses the Cancellation API

**Tests**: 6 tests

---

## 2026-07-16 — CLI Tool

**What**: Basic command-line interface for common GraphForge operations.

**Changes**:
- ``graphforge/cli.py`` — new module with CLI using ``argparse``.

**Commands**:
- ``graphforge run <graph.json> <state.json>`` — invoke a serialized graph
- ``graphforge viz <graph.json>`` — export graph visualization
- ``graphforge info <graph.json>`` — show graph topology info

**Tests**: 4 tests (argparse parsing, mock runs)

---

## 2026-07-16 — State Middleware

**What**: Pre- and post-processing hooks for state transitions, allowing
global interceptors for logging, validation, and transformation.

**Changes**:
- ``graphforge/_middleware.py`` — new module with ``StateMiddleware``
  Protocol and ``MiddlewarePipeline``.
- ``graphforge/_executor.py`` — Executor calls middleware hooks before
  and after each state update.

**Design**:
- ``StateMiddleware.pre_update(node, state, updates) -> updates``
- ``StateMiddleware.post_update(node, old_state, new_state) -> None``
- ``MiddlewarePipeline(stages: List[StateMiddleware])`` composes multiple
  middlewares in order.
- Middleware can modify updates before they are applied (e.g., logging,
  validation, encryption).

**Tests**: 5 tests

---


---

## 2026-07-16 — v0.4.0 Feature Implementation

**What**: Final round of feature implementation focused on three P0 capabilities:
Embeddings/Vector Search/RAG, Cost/Token tracking, and Human-in-the-loop
approval patterns.

---

## 2026-07-16 — Embeddings / Vector Search / RAG Module

**What**: Added a complete RAG (Retrieval-Augmented Generation) module enabling
agents to retrieve knowledge from vector databases.

**Changes**:
- ``graphforge/rag/`` — new module with ``Embeddings`` ABC, ``VectorStore`` ABC,
  ``InMemoryVectorStore``, ``RetrievalNode``, and chunking utilities.
- ``graphforge/rag/_embeddings.py`` — ``Embeddings`` abstract base class
- ``graphforge/rag/_store.py`` — ``VectorStore`` ABC + ``InMemoryVectorStore``
- ``graphforge/rag/_node.py`` — ``RetrievalNode`` for use in graphs
- ``graphforge/rag/_chunking.py`` — Text chunking utilities
- ``graphforge/rag/__init__.py`` — public API

**Design**:
- ``Embeddings`` ABC with ``embed_documents()`` and ``embed_query()`` methods
- ``VectorStore`` ABC with ``add_texts()``, ``similarity_search()``
- ``InMemoryVectorStore`` uses cosine similarity with numpy
- ``RetrievalNode`` is a first-class graph node that retrieves context into state
- Chunking supports fixed-size, recursive, and sentence-based strategies

**Tests**: 15 tests

---

## 2026-07-16 — Cost / Token Tracking

**What**: Built-in token usage and cost tracking via the callback system.

**Changes**:
- ``graphforge/_callbacks.py`` — Added ``CostCallback`` that tracks per-node
  token usage and computes costs based on model pricing tables.
- ``CostCallback.track(model, prompt_tokens, completion_tokens)`` API
- Built-in pricing table for common models (GPT-4, GPT-3.5, Claude, etc.)

**Design**:
- ``CostCallback.get_stats()`` returns per-node and total costs
- ``CostCallback.total_cost()`` returns aggregate cost
- Custom pricing via ``CostCallback.set_pricing(model, input_price, output_price)``
- Compatible with ``CallbackManager``

**Tests**: 8 tests

---

## 2026-07-16 — Human-in-the-Loop Approval Patterns

**What**: Enhanced interrupt/resume with approval workflows, timeouts, and
decision handling.

**Changes**:
- ``graphforge/_interrupt.py`` — Enhanced ``interrupt()`` with ``timeout``
  and ``on_timeout`` parameters.
- ``graphforge/agents/patterns.py`` — Added ``ApprovalNode`` factory for
  creating approval-required graph nodes.

**Design**:
- ``interrupt(timeout=..., on_timeout="reject")`` — interrupt with configurable
  timeout behavior
- ``ApprovalNode(fn, timeout=..., on_timeout=...)`` — wraps any node function
  with an approval gate before execution
- ``on_timeout="reject"`` (auto-reject), ``"approve"`` (auto-approve),
  ``"raise"`` (raise error)
- Resume passes approval decision via ``resume(updates={"decision": "approve"})``

**Tests**: 8 tests

---


---

## 2026-07-16 — v0.5.0 — 逆天功能实现

**What**: Three industry-first features that go beyond LangGraph's capabilities:
unified ``serve()``, automatic graph optimization, and execution timeline debugging.

---

## 2026-07-16 — serve() — One-Command Unified API Server

**What**: Added a single ``serve()`` function that converts any CompiledGraph
into a production-ready API server supporting REST, WebSocket, MCP, A2A, and
Swagger documentation — simultaneously.

**Changes**:
- ``graphforge/serve.py`` — new module with ``serve()`` function
- Unifies ``GraphServer`` (HTTP/WS), ``MCPAgentServer``, and ``A2AServer``

**Design**:
- ``serve(graph, host="0.0.0.0", port=8080)`` starts all protocols on one port
- HTTP: POST /invoke, POST /stream (SSE), GET /health
- WebSocket: /ws for bidirectional streaming
- MCP: /.well-known/mcp for tool discovery, /mcp/call for invocation
- A2A: /.well-known/agent-card, POST /tasks/send
- Swagger: GET /docs with auto-generated OpenAPI spec
- Auth: Optional API key support

**Tests**: 6 tests (import, structure, config)

---

## 2026-07-16 — AutoOptimizer — Automatic Graph Parallelization

**What**: Static graph analysis that automatically detects independent execution
paths and converts them to parallel fan-out edges, without user annotation.

**Changes**:
- ``graphforge/_optimizer.py`` — new module with ``AutoOptimizer`` class

**Design**:
- Analyzes node dependencies based on state field read/write patterns
- Detects independent subgraphs (nodes that don't depend on each other)
- Automatically inserts fan-out edges for independent paths
- ``auto_parallelize(graph, state_type)`` returns an optimized graph
- Also detects: unused nodes, redundant computations, potential bottlenecks
- ``optimize(graph, state_type)`` returns ``OptimizationReport`` with suggestions

**Tests**: 8 tests

---

## 2026-07-16 — TimelineRecorder — Execution Recording & Replay

**What**: Full execution recording that captures every state transition for
post-mortem debugging, replay, and inspection.

**Changes**:
- ``graphforge/_timeline.py`` — new module with ``TimelineRecorder``

**Design**:
- Records every state transition with full context (node, updates, timing)
- ``TimelineRecorder.get_timeline()`` returns ordered list of frames
- ``TimelineRecorder.export_json()`` for offline analysis
- ``TimelineRecorder.replay()`` yields frames in order for custom replay
- Compatible with existing ``Callback`` protocol
- Frames include: node name, step, state before, state after, updates, duration

**Tests**: 8 tests

---


---

## 2026-07-16 — v0.6.0 — 最终轮: ReAct 流式, 多模态, Agent UI, 分布式

---

## 2026-07-16 — ReAct 循环内流式

**What**: ToolNode 和 ReAct agent 现在支持 generator 形式的 LLM 函数，
实现 token 级流式输出。

**Changes**:
- ``graphforge/agents/_tool_node.py`` — ToolNode 检测 LLM 函数是否为
  generator，若是则逐个 token 处理
- ``graphforge/agents/_react.py`` — create_react_agent 支持流式模式

**Design**:
- LLM 函数可以是 ``def llm(messages, tools): yield token``
- ToolNode 自动检测并逐 token 处理
- 与现有的 EventType.STREAM_TOKEN 集成
- 兼容非流式 LLM 函数（向下兼容）

**Tests**: 6 tests

---

## 2026-07-16 — 多模态支持 (Image)

**What**: 新增多模态模块，支持 agent 处理图片输入和生成图片输出。

**Changes**:
- ``graphforge/multimodal/`` — 新模块
- ``graphforge/multimodal/_image.py`` — ImageNode 支持图片输入/输出
- ``graphforge/multimodal/__init__.py`` — 公开 API

**Design**:
- ``ImageNode`` — 图片处理节点：base64 输入/输出，URL 加载，尺寸调整
- 兼容 OpenAI 的 image_url 格式（多模态 LLM）
- 无外部依赖（标准库 PIL 可选增强）

**Tests**: 6 tests

---

## 2026-07-16 — Auto-Generated Agent Web UI

**What**: 任意 CompiledGraph 自动生成可交互的 Web Dashboard，
显示图结构、节点状态、实时执行。

**Changes**:
- ``graphforge/_dashboard.py`` — 内嵌 HTML/CSS/JS 的 Web Dashboard
- ``graphforge/_http_server.py`` — GraphServer 添加 GET /dashboard 端点

**Design**:
- Python 生成单个 HTML 页面（无外部依赖）
- 使用现有 REST API 执行 graph
- Mermaid.js 渲染图拓扑（CDN 加载）
- 实时显示执行状态和结果
- 完全的暗色主题（匹配 logo 风格）

**Tests**: 3 tests

---

## 2026-07-16 — 分布式执行 (Thread Pool)

**What**: 分布式图执行支持，允许节点在远程 worker 上执行。

**Changes**:
- ``graphforge/distributed.py`` — 新模块，基于 concurrent.futures 的
  分布式执行器

**Design**:
- ``DistributedExecutor`` — 将节点分发到远程 worker 执行
- 支持本地线程池、进程池、Future-based 远程执行
- 兼容现有的 Checkpointer 和 Store
- 零额外依赖（标准库 concurrent.futures）

**Tests**: 4 tests

---


---

## Design Principles Applied

Throughout these improvements, the following principles guided the work:

1. **Type safety first**: All new APIs use generics and Protocols
2. **Explicit over implicit**: Configuration is explicit (retry, timeout, maps)
3. **Composability**: New features compose with existing graph patterns
4. **Minimal surface area**: New modules are optional (a2a, agents)
5. **Test coverage**: Every feature has targeted tests
