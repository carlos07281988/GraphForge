# GraphForge Examples

Quickstart examples organized by complexity and feature area.

## Quick Start

```bash
cd ../
pip install -e .  # or: PYTHONPATH=$PWD:$PYTHONPATH
cd examples
python3 00_quick_start.py
```

If you haven't installed graphforge, use:

```bash
PYTHONPATH=..:$PYTHONPATH python3 00_quick_start.py
```

## Example Index

| # | File | Feature | Run |
|---|---|---|---|
| 00 | `00_quick_start.py` | Minimal graph — define state, nodes, edges, compile, invoke | `python3 00_quick_start.py` |
| 01 | `01_state_management.py` | Merge strategies: overwrite, append, reduce + immutability | `python3 01_state_management.py` |
| 02 | `02_conditional_routing.py` | Dynamic routing with conditional edges | `python3 02_conditional_routing.py` |
| 03 | `03_streaming.py` | Token-level streaming, event streaming, value streaming | `python3 03_streaming.py` |
| 04 | `04_react_agent.py` | ReAct agent with tool calling (manual + helper) | `python3 04_react_agent.py` |
| 05 | `05_multi_agent.py` | Supervisor/Worker and Swarm orchestration patterns | `python3 05_multi_agent.py` |
| 06 | `06_rag.py` | RAG with vector store, embeddings, retrieval node | `python3 06_rag.py` |
| 07 | `07_guardrails.py` | Input/output guardrails, PII filtering, length limits | `python3 07_guardrails.py` |
| 08 | `08_checkpointing.py` | Checkpoint save/restore across sessions | `python3 08_checkpointing.py` |
| 09 | `09_serve.py` | Deploy graph as REST API with serve() | `python3 09_serve.py` |
| 10 | `10_store_memory.py` | Long-term memory via Store injection | `python3 10_store_memory.py` |
| 11 | `11_cost_tracking.py` | Token usage and cost tracking | `python3 11_cost_tracking.py` |
| 12 | `12_complex_graph.py` | Combining streaming, routing, store, memory, timing | `python3 12_complex_graph.py` |

## What Each Example Demonstrates

### 00-03: Core Concepts
Basic graph construction, state management, routing, and streaming.

### 04-05: Agent Patterns
ReAct agents with tool calling, multi-agent orchestration patterns.

### 06-07: Knowledge & Safety
RAG with vector search, guardrails for input/output validation.

### 08-10: Persistence & Deployment
Checkpointing, API server deployment, long-term memory.

### 11-12: Production Features
Cost tracking, comprehensive graph combining all features.

## Tips

- All examples are self-contained — no external API keys needed
- Examples use `configure_logging()` — set `level=logging.DEBUG` for verbose output
- Examples 04/06 use simulated LLMs — replace with real API calls for production
- Run examples from the `examples/` directory
