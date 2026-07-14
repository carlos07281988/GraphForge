<div align="center">

<pre width="80">
   ________                       __     ____             __
  / ____/  _____  ____   ____ _  / /__  / __/  ____ ___  / /_
 / / __   / _ \ / __ \ / __ `/ / //_/ / /_   / __// _  |/ __/
/ /_/ /  /  __// /_/ // /_/ / / ,<   / __/ _/ /_ / ,__// /_
\____/   \___/ \___/ \__, / /_/|_| /_/   (_)___//_/    \__/
                    /____/
</pre>

**GraphForge** · _类型安全、可组合的图执行框架，专为 LLM 应用设计_

<br/>

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?style=flat&logo=python)](https://www.python.org)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-4A90D9?style=flat&logo=python)](https://docs.pydantic.dev)
[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat)](LICENSE)

<br/>

</div>

GraphForge 是一个用于构建**有状态、可组合执行图**的 Python 框架，专为 LLM 驱动应用设计。它借鉴了 [LangGraph](https://github.com/langchain-ai/langgraph) 的状态图模型，从底层起就**强制类型安全**，让**状态管理显式化**，并保持**最小的抽象表面积**——同时刻意避免 LangGraph 和 LangChain 的设计缺陷。

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

## 目录

- [为什么选择 GraphForge？](#为什么选择-graphforge)
- [安装](#安装)
- [核心概念](#核心概念)
  - [GraphState](#graphstate)
  - [节点 (Nodes)](#节点-nodes)
  - [边 (Edges)](#边-edges)
  - [Graph & CompiledGraph](#graph--compiledgraph)
  - [条件路由](#条件路由)
  - [Pipeline](#pipeline)
  - [节点级重试与错误回退](#节点级重试与错误回退)
  - [子图输入/输出映射](#子图输入输出映射)
  - [A2A（Agent-to-Agent）协议](#a2aagent-to-agent协议)
  - [智能体（ToolNode + ReAct）](#智能体toolnode--react)
- [执行模式](#执行模式)
  - [Invoke](#invoke)
  - [Streaming](#streaming)
  - [Async](#async)
- [日志](#日志)
- [检查点 (Checkpointing)](#检查点-checkpointing)
- [回调 (Callbacks)](#回调-callbacks)
- [API 参考](#api-参考)
- [架构](#架构)
- [与 LangGraph 对比](#与-langgraph-对比)
- [Roadmap](#roadmap)

---

## 为什么选择 GraphForge？

LangGraph 带来了强大的思维模型——有状态图用于智能体工作流——但其实现存在以下问题：

- **混乱的状态管理**：TypedDict + 神奇的 ``__reducers__``，需要在运行时解析注解。
- **弱类型安全**：节点之间没有端到端的类型保障。
- **不透明的内部机制**：基于通道的消息传递让调试变得困难。
- **可变状态**：节点可能意外地在步骤间泄漏副作用。

GraphForge 走**第三条路**：保留图模型，从底层起强制类型安全，让状态管理显式化，最小化抽象表面积。

| 问题 | GraphForge 的解决方案 |
|---|---|
| 状态模式 | Pydantic v2——可验证、可序列化、可自省 |
| 合并语义 | 字段级 `node_field(merge=...)`——每个字段显式声明 |
| 类型安全 | `Generic[StateT]` 贯穿每个抽象层 |
| 不可变性 | 每一步 = `model_copy(update=..., deep=True)`——从不原地修改 |
| 图组合 | 子图和 Pipeline 作为一等公民节点 |

---

## 安装

```bash
pip install graphforge
```

**要求**：Python 3.9+、Pydantic v2、typing_extensions。

---

## 核心概念

### GraphState

状态是流经图的数据。将其定义为 Pydantic v2 的 `BaseModel`，并为每个字段声明合并语义：

```python
from graphforge import GraphState, node_field

class AgentState(GraphState):
    # 覆盖语义(默认)：每个节点替换旧值
    status: str = "idle"

    # 追加语义：节点更新被扩展到现有列表中
    messages: list[dict] = node_field(default=[], merge="append")

    # 自定义 reducer：(old, new) -> value
    total_tokens: int = node_field(
        default=0,
        merge="reduce",
        reducer=lambda old, new: (old or 0) + new,
    )
```

**合并策略**：

| 策略 | 行为 | 典型用途 |
|---|---|---|
| `overwrite` | 旧值被新值替换 | 状态标志、标量字段 |
| `append` | 新项目追加到列表 | 消息历史、路径追踪 |
| `reduce` | 自定义 `(old, new) -> value` | 累加器、计数器 |

使用 `state.apply(**updates)` 产生新的快照：

```python
state = AgentState()
state2 = state.apply(status="running")          # overwrite
state3 = state2.apply(messages=Append([msg]))    # append
```

**状态是不可变的**——`apply()` 总是返回新实例。

### 节点 (Nodes)

节点是一个可调用对象 `(state, **kwargs) -> dict`。它接收当前状态并返回要更改的字段：

```python
def llm_call(state: AgentState) -> dict:
    response = call_llm(state.messages)
    return {
        "messages": Append([{"role": "assistant", "content": response}]),
        "total_tokens": response.tokens,
    }
```

节点也可以是：
- **异步函数**：`async def node(state): ...`
- **生成器**：`def node(state): yield update1; yield update2`
- **CompiledGraph**：将子图嵌入为节点
- **Pipeline**：将线性管道嵌入为节点

### 边 (Edges)

边连接节点。有两种类型：

```python
# 无条件边：a 总是到 b
graph.add_edge("a", "b")

# 条件边：router 根据状态决定目标
graph.add_conditional_edges(
    "classify",
    router=lambda s: "process" if s.has_data else "wait",
    path_map={"process": "processor", "wait": "input_handler"},
)
```

### Graph & CompiledGraph

使用可变的 `Graph` 构建器构建图，然后编译：

```python
graph = (
    Graph[AgentState]()
    .add_node("llm", llm_call)
    .add_node("tools", tool_executor)
    .add_edge("llm", "tools")
    .add_edge("tools", "__end__")      # __end__ 是终止哨兵
    .set_entry_point("llm")
    .compile(name="my_agent")
)
```

`CompiledGraph` 是不可变的。一旦编译完成，拓扑结构再也不会改变。

### 条件路由

Router 检查状态并返回 `path_map` 中的键：

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

从条件边终止，让 `path_map` 返回 `"__end__"`。

### 并行分支 (Fan-out)

使用 `add_fanout()` 将节点并发分发到多个分支：

```python
graph.add_fanout("classify", ["search", "calculate"], join="synthesize")
```

- **Sync 模式**：分支按顺序执行
- **Async 模式**：分支通过 `asyncio.gather` 并发执行
- **Join**：指定汇合节点，所有分支完成后执行

### Pipeline

`Pipeline` 是一个线性步骤序列（无分支）。也可以作为节点嵌入到图中：

```python
from graphforge import Pipeline

pipe = Pipeline[AgentState]([
    step_one,
    step_two,
], name="preprocess")

# 作为图节点使用
graph.add_node("preprocess", pipe)
```


### 节点级重试与错误回退

节点可配置自动重试，图可在节点重试耗尽后路由到备用节点。

**重试：** ``add_node()`` 传递 ``retry=N``：

```python
def flaky_node(state) -> dict:
    return {"x": 1}

graph.add_node("不稳定节点", flaky_node, retry=3)
```

执行器会最多重试 ``retry + 1`` 次。

**错误边：** 使用 ``add_error_edge()`` 定义备用路径：

```python
graph.add_node("主节点", flaky_node)
graph.add_node("备用节点", fallback)
graph.add_error_edge("主节点", "备用节点")
```

当主节点异常且重试耗尽后，执行自动转到备用节点。


### 子图输入/输出映射

将子图嵌入为节点时，可以通过 ``input_map`` 和 ``output_map`` 声明父状态与子图状态之间的映射。

```python
class 父状态(GraphState):
    query: str = ""
    result: str = ""

class 子状态(GraphState):
    prompt: str = ""
    output: str = ""

sub = (
    Graph[子状态]()
    .add_node("处理", lambda s: {"output": f"结果: {s.prompt}"})
    .add_edge("处理", "__end__")
    .set_entry_point("处理")
    .compile(
        state_type=子状态,
        input_map={"query": "prompt"},
        output_map={"output": "result"},
    )
)

parent = Graph[父状态]().add_node("sub", sub).add_edge("sub", "__end__").set_entry_point("sub").compile()
result = parent.invoke(父状态(query="hello"))
print(result.result)  # "结果: hello"
```

### A2A（Agent-to-Agent）协议

GraphForge 内置了对 Google [Agent-to-Agent (A2A)](https://google.github.io/A2A/) 开放协议的支持，
让基于 GraphForge 构建的 agent 能够与任何其他框架的 agent 互相通信。

A2A 模块提供**出站**和**入站**两个方向的集成：

| 方向 | 机制 | 使用场景 |
|---|---|---|
| **出站**（GraphForge → 远程 agent） | `create_a2a_agent_node()` | 在图内调用第三方 agent |
| **入站**（远程 agent → GraphForge） | `A2AServer` | 把你的图暴露为标准 A2A 端点 |

**安装：**

```bash
pip install graphforge[a2a]   # 会自动安装 aiohttp
```

#### 出站：调用外部 Agent

```python
from graphforge.a2a import create_a2a_agent_node

# 创建一个委托给远程 A2A agent 的节点
get_weather = create_a2a_agent_node("http://weather-agent:8080")

# 像普通节点一样使用
graph.add_node("get_weather", get_weather)
graph.add_edge("user_input", "get_weather")
```

通过自定义映射器（mapper）控制图状态与 A2A 消息之间的转换：

```python
def my_input(state) -> Message:
    return Message(role="user", parts=[TextPart(text=state.prompt)])

def my_output(msg, task) -> dict:
    text = msg.parts[0].text if msg and msg.parts else "done"
    return {"messages": Append([{"role": "assistant", "content": text}])}

node = create_a2a_agent_node(
    "http://agent:8080",
    input_mapper=my_input,
    output_mapper=my_output,
)
```

#### 入站：将 Graph 暴露为 A2A Agent

```python
from graphforge.a2a import A2AServer, AgentCard, AgentSkill

card = AgentCard(
    name="SupportBot",
    description="客服 agent",
    capabilities=AgentCapabilities(
        skills=[AgentSkill(id="triage", name="问题分类")],
    ),
)

server = A2AServer(
    compiled_graph,
    agent_card=card,
    host="0.0.0.0",
    port=8080,
)

server.run()                    # 阻塞启动
# 或异步启动/停止
await server.start()
await server.stop()
```

启动后，任何兼容 A2A 的 agent 都可以通过以下端点发现并调用你的图：

```
GET  /.well-known/agent-card     # 服务发现
POST /tasks/send                  # 同步任务
POST /tasks/sendStream            # 流式任务（SSE）
GET  /tasks/{id}                  # 任务状态查询
POST /tasks/{id}/cancel           # 取消任务

---

### 智能体（ToolNode + ReAct）

GraphForge 内置了智能体模式和工具调用支持，位于 ``graphforge.agents`` 模块。

**ToolNode** — 调用 LLM、执行工具调用、将结果追加到消息列表的节点：

```python
from graphforge.agents import ToolNode

tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索",
        },
        "_func": search,
    }
]

graph.add_node("agent", ToolNode(llm, tools=tools))
```

**路由** — ``has_tool_calls()`` 检查最后一条消息是否包含工具调用：

```python
graph.add_conditional_edges("agent", has_tool_calls, {
    "tools": "tools",
    "end": "__end__",
})
```

**ReAct 智能体** — ``create_react_agent()`` 一键构建完整的推理+行动循环：

```python
graph = create_react_agent(llm, tools=tools)
compiled = graph.compile(state_type=ReactState)
result = compiled.invoke(ReactState(messages=[{"role": "user", "content": "搜索 AI 新闻"}]))
```


## 执行模式

### Invoke

```python
result = compiled.invoke(initial_state, config={
    "thread_id": "session-123",
    "recursion_limit": 50,
})
```

### Streaming

流式执行逐步骤产生事件，适用于实时界面或进度监控：

```python
from graphforge import EventType

for event in compiled.stream(initial_state):
    if event.type == EventType.NODE_START:
        print(f"⏳ {event.node}")
    elif event.type == EventType.STATE_UPDATE:
        print(f"  ├ 更新: {list(event.data['updates'].keys())}")
    elif event.type == EventType.GRAPH_END:
        print(f"✅ 完成: {event.data}")
```

### Async

```python
import asyncio

# 异步调用
result = asyncio.run(compiled.ainvoke(initial_state))

# 异步流式
async for event in compiled.astream(initial_state):
    print(event)
```

`AsyncExecutor` 透明地处理同步和异步节点。

---

## 日志

GraphForge 在所有模块中包含结构化日志。

### 快速设置

```python
from graphforge import configure_logging
import logging

configure_logging(level=logging.DEBUG)
```

输出样例：

```
15:42:01 [graphforge.graph] INFO  编译图: 3 个节点, 4 条边, 1 条条件边
15:42:01 [graphforge.executor] INFO  图 "my_agent" 启动 (thread="default", entry="classify")
15:42:01 [graphforge.executor] INFO  节点 "classify" (step=0, kind=function)
15:42:01 [graphforge.executor] INFO  图 "my_agent" 在 2 步后完成
```

### 模块日志器

| 日志器 | 模块 | 级别 |
|---|---|---|
| `graphforge.graph` | 图构建与编译 | INFO, DEBUG |
| `graphforge.node` | 节点分发 | DEBUG |
| `graphforge.state` | 状态合并 | DEBUG |
| `graphforge.executor` | 执行引擎 | INFO, DEBUG, ERROR |
| `graphforge.checkpoint` | 检查点 | DEBUG |
| `graphforge.callback` | 回调分发 | DEBUG |
| `graphforge.pipeline` | Pipeline 执行 | INFO, DEBUG |

---

## 检查点 (Checkpointing)

检查点在每个节点执行后保存状态，支持恢复和调试：

```python
from graphforge import InMemoryCheckpointer

checkpointer = InMemoryCheckpointer()
compiled = graph.compile(checkpointer=checkpointer)
compiled.invoke(initial_state)

# 检查执行历史
for key in checkpointer.list("default"):
    snapshot = checkpointer.get(key)
    print(f"  {key[0]} / {key[1]} / step {key[2]}")
```

### 恢复执行

```python
# 从最后一个检查点恢复
result = compiled.resume("session-1", state_type=MyState)

# 带状态更新恢复（如人工输入）
result = compiled.resume("session-1", state_type=MyState, updates={"input": "yes"})
```

### 持久化存储

```python
from graphforge import SqliteCheckpointer, RedisCheckpointer

# SQLite 持久化
cp = SqliteCheckpointer("checkpoints.db")

# Redis 分布式存储
import redis
cp = RedisCheckpointer(redis.Redis(), key_prefix="gf:")
```

---

## 回调 (Callbacks)

回调让你无需修改节点即可挂接到执行过程：

```python
from graphforge import Callback, CallbackManager

class Logger(Callback):
    def on_node_start(self, node: str, state: dict) -> None:
        print(f"  开始 {node}")

    def on_node_end(self, node: str, state: dict) -> None:
        print(f"  完成 {node}")

manager = CallbackManager([Logger()])
compiled.invoke(state, callbacks=manager)
```

**可用钩子**：

| 钩子 | 调用时机 |
|---|---|
| `on_graph_start(graph_name, input_state)` | 执行开始 |
| `on_graph_end(graph_name, final_state)` | 执行完成 |
| `on_graph_error(graph_name, error)` | 未处理的异常 |
| `on_node_start(node, state)` | 节点运行前 |
| `on_node_end(node, state)` | 节点成功后 |
| `on_node_error(node, error)` | 节点抛出异常 |
| `on_state_update(node, updates, new_state)` | 状态合并后 |
| `on_conditional_edge(node, result, target)` | Router 求值后 |

---

## API 参考

### `Graph[StateT]`

| 方法 | 返回 | 描述 |
|---|---|---|
| `add_node(name, fn, *, retry=0, timeout=None, metadata=None)` | `Self` | 注册节点（支持重试） |
| `add_edge(source, target)` | `Self` | 无条件边 |
| `add_error_edge(source, fallback)` | `Self` | 错误回退边 |
| `add_conditional_edges(source, router, path_map)` | `Self` | 条件路由 |
| `add_fanout(source, targets, join=None)` | `Self` | 并行分支 |
| `set_entry_point(name)` | `Self` | 设置起始节点 |
| `set_finish_point(name)` | `Self` | 设置终止节点 |
| `set_metadata(key, value)` | `Self` | 附加元数据 |
| `compile(*, input_map=None, output_map=None, checkpointer=None, name=None, state_type=None)` | `CompiledGraph` | 冻结并验证 |

### `CompiledGraph[StateT]`

| 方法 | 返回 | 描述 |
|---|---|---|
| `invoke(state, config=None, callbacks=None)` | `StateT` | 同步执行 |
| `ainvoke(state, config=None, callbacks=None)` | `StateT` | 异步执行 |
| `stream(state, config=None, callbacks=None)` | `Generator[StreamEvent]` | 同步流式 |
| `astream(state, config=None, callbacks=None)` | `AsyncGenerator[StreamEvent]` | 异步流式 |
| `resume(thread_id, state_type=None, updates=None, ...)` | `StateT` | 从检查点恢复 |
| `aresume(thread_id, state_type=None, updates=None, ...)` | `StateT` | 异步恢复 |
| 属性：`name`, `nodes`, `entry_point`, `finish_points`, `checkpointer`, `metadata`, `state_type`, `input_map`, `output_map`, `error_map` | | 只读 |

### 图可视化

```python
from graphforge import export_dot, render_graph

# 导出 DOT 格式
dot = export_dot(compiled, show_kind=True)

# 渲染为图片（需要 graphviz 包）
render_graph(compiled, "graph.png")
```

---

## 架构

深入设计细节请参阅：

[`docs/architecture.md`](docs/architecture.md) — 英文版
[`docs/architecture.zh-CN.md`](docs/architecture.zh-CN.md) — 中文版

涵盖内容：

- 模块依赖图
- 状态合并管道
- 执行生命周期（同步、异步、流式）
- 检查点模型
- 设计决策与理由
- 扩展点
- 与 LangGraph 和 LangChain 的完整对比

---

## 与 LangGraph 对比

| 领域 | LangGraph | GraphForge |
|---|---|---|
| 状态模型 | TypedDict + 神奇的 ``__reducers__`` | Pydantic v2 + 显式的 `node_field(merge=...)` |
| 类型安全 | 弱（TypedDict 运行时脆弱） | 完整泛型、Protocols、静态分析 |
| 不可变性 | 原地修改状态 | 每一步 = 新的 `model_copy` 快照 |
| 图组合 | 基于通道的内部机制 | 子图和 Pipeline 作为一等公民节点 |
| 流式 | 不透明的通道事件 | 每步 `StreamEvent` 对象 |
| 异步支持 | 有异步 API | 从第一天起原生同步+异步 |
| 检查点 | `BaseCheckpointSaver` | `Checkpointer` ABC + 内存/SQLite/Redis |
| 回调 | 无 | `Callback` Protocol + `CallbackManager` |
| 依赖 | langchain-core, pydantic, 更多 | 仅 Pydantic v2, typing_extensions |
| 错误处理 | 让未处理异常传播 | 日志 + 分发给回调 + 重新抛出 |

---

所有改进的详细记录请参阅 [`docs/improvements.md`](docs/improvements.md)。

## Roadmap

- [x] `resume()` API — 基于检查点的恢复，支持暂停/重试
- [x] SQLite 检查点 — 持久化状态存储，完整 CRUD，线程安全
- [x] 并行/扇出节点执行 — `add_fanout()` API，异步并行，join 支持
- [x] 子图检查点隔离 — 自动线程 ID 前缀，共享检查点支持
- [x] 图可视化 — `export_dot()` DOT 导出，`render_graph()` 图片渲染
- [x] Redis 检查点 — 分布式持久化
- [x] Pydantic v1 兼容 — v1/v2 API 统一兼容层
- [x] A2A（Agent-to-Agent）协议 — 出站/入站 agent 通信
- [x] 节点级重试与错误回退 — retry=N, add_error_edge()
- [x] 子图 I/O 映射 — input_map/output_map 父子边界声明
- [x] 智能体模块 — ToolNode、has_tool_calls()、create_react_agent()
- [x] 图序列化 — serialize()/deserialize() JSON/YAML 导出/导入

### 未来工作

- A2A 推送通知（基于 Webhook 的任务更新）（基于 Webhook 的任务更新）
- OpenTelemetry 追踪（节点级可观测性）
- 人机协同模式（审批节点、中断/恢复）
- 分布式执行（Dask/Ray 集成）

---

## 许可证

Apache 2.0 — 参见 [`LICENSE`](LICENSE)。
