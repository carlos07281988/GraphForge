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

## Design Principles Applied

Throughout these improvements, the following principles guided the work:

1. **Type safety first**: All new APIs use generics and Protocols
2. **Explicit over implicit**: Configuration is explicit (retry, timeout, maps)
3. **Composability**: New features compose with existing graph patterns
4. **Minimal surface area**: New modules are optional (a2a, agents)
5. **Test coverage**: Every feature has targeted tests
