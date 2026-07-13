# GraphForge 架构文档

> **版本**: 0.1.0
> **状态**: 设计参考
> **最后更新**: 2026-07-13

## 目录

1. [设计哲学](#1-设计哲学)
2. [核心抽象](#2-核心抽象)
3. [模块依赖图](#3-模块依赖图)
4. [状态管理深入](#4-状态管理深入)
5. [图执行生命周期](#5-图执行生命周期)
6. [流式架构](#6-流式架构)
7. [检查点模型](#7-检查点模型)
8. [对比](#8-对比)
9. [设计决策与理由](#9-设计决策与理由)
10. [扩展点](#10-扩展点)

---

## 1. 设计哲学

GraphForge 建立在一组明确的设计约束之上，使其既区别于 LangGraph，也区别于 LangChain。

### 1.1 从底层起类型安全

每个公共 API 接口都携带完整的泛型类型参数。`StateT` TypeVar 流经节点、边、图、执行器和检查点，实现端到端的类型检查而无需强制转换。这消除了一整类困扰那些依赖鸭子类型字典的框架的运行时错误。

### 1.2 显式状态契约

状态模式是 Pydantic v2 的 `BaseModel` 子类。每个字段通过 `node_field(merge=...)` 显式声明其合并策略。没有隐式的 reducer，没有类级别的 `__reducers__` 魔法，也没有在运行时自省 TypedDict 注解。节点和框架之间的契约是：*"我返回一个我想更改的字段字典；你来决定如何合并它们。"*

### 1.3 不可变快照

每个调用步骤都通过 `model_copy(update=..., deep=True)` 产生一个新的状态实例。框架从不原地修改状态。这提供了：

- **确定性**：用相同的输入和检查点重放图总是产生相同的执行路径。
- **可调试性**：每个状态快照都作为检查点保存；你可以在任何步骤检查状态。
- **安全性**：节点不能意外地在步骤间泄漏突变。

### 1.4 最小表面积

框架正好导出了 7 个公共抽象（Graph, CompiledGraph, GraphState, Node, Pipeline, StreamEvent, Checkpointer）加上支持类型。没有 `BaseLLM`，没有 `Chain`，没有 `Toolkit`，没有 `Memory`。这些属于用户代码或配套库，而非核心运行时。

### 1.5 组合优于继承

图通过子图嵌入（`CompiledGraph` 可用作另一个图中的节点）和 Pipeline 嵌入（`Pipeline` 可用作节点）来组合。没有需要继承的类层次结构——只是消费状态并返回更新的可调用对象。

### 1.6 透明执行

所有执行路径（同步、异步、流式）都会发出结构化事件，可通过回调或流迭代器消费。没有隐藏的通道协议或不透明的内部队列。

---

## 2. 核心抽象

### 2.1 GraphState

```
GraphState (Pydantic BaseModel)
  ├── apply(**updates) -> Self     # 不可变合并入口
  ├── model_dump() -> dict          # Pydantic 序列化
  └── model_copy(update=...)        # 快照机制
```

每个字段可以携带通过 `json_schema_extra` 附加的 `ReducerDescriptor`，配置如何折叠更新：

| 策略 | 行为 | 用例 |
|---|---|---|
| `overwrite` (默认) | 用新值替换旧值 | 标量字段（名称、计数、标志） |
| `append` | 扩展现有列表 | 消息历史、路径追踪 |
| `reduce` | 调用 `(old, new) -> value` | 累加器、自定义合并逻辑 |

### 2.2 Node

```
Node[StateT]
  ├── name: str
  ├── kind: NodeKind (FUNCTION | ASYNC | STREAM | SUBGRAPH | PIPELINE)
  ├── invoke(state) -> dict         # 同步执行
  ├── ainvoke(state) -> dict        # 异步执行
  └── stream(state) -> Generator    # 流式执行
```

`Node` 包装任何可调用对象并在构造时进行分类。分类（由 `_classify()` 执行）使用 `inspect` 检查可调用对象，确定它是同步、异步、生成器、`CompiledGraph` 还是 `Pipeline`。这使得执行器能够正确分发，而无需在每次调用时进行运行时类型检查。

### 2.3 Graph

```
Graph[StateT] (可变构建器)
  ├── add_node(name, fn)            # 注册节点
  ├── add_edge(source, target)      # 无条件边
  ├── add_conditional_edges(src, router, path_map)  # 条件路由
  ├── add_fanout(source, targets, join)  # 并行扇出分支
  ├── set_entry_point(name)         # 定义起始节点
  ├── set_finish_point(name)        # 定义终止节点
  └── compile(...) -> CompiledGraph # 冻结并验证
```

`Graph` 故意设计为可变的（构建器模式），而 `CompiledGraph` 故意设计为不可变的。这种分离：
1. 允许任意顺序的增量构建。
2. 在编译时启用验证（悬空边、缺少入口点、无效的路由器目标）。
3. 保证编译后的图的拓扑结构在执行期间永远不会改变。

### 2.4 CompiledGraph

```
CompiledGraph[StateT] (不可变)
  ├── invoke(state) -> state        # 同步执行
  ├── ainvoke(state) -> state       # 异步执行
  ├── stream(state) -> StreamEvent  # 流式执行
  ├── astream(state) -> StreamEvent # 异步流式
  ├── resume(thread_id, state_type, updates)  # 从检查点恢复
  ├── nodes, entry_point, finish_points, checkpointer, state_type
  └── successors(name) -> [NodeName]
```

编译后的图在构造时预先计算后继表和条件边查找映射，实现执行期间的 O(1) 分发。

### 2.5 Pipeline

```
Pipeline[StateT]
  ├── run(state) -> dict            # 同步顺序执行
  ├── arun(state) -> dict           # 异步顺序执行
  └── steps: [Callable]
```

`Pipeline` 是一个线性可调用对象序列。与 `Graph` 不同，它没有分支、没有路由、没有循环。每个步骤接收所有先前步骤的*累积*输出。Pipeline 可以通过 `Graph.add_node(name, pipeline)` 作为节点嵌入到图中。

### 2.6 边类型

```
DirectEdge[StateT]                 ConditionalEdge[StateT]           FanOutEdge[StateT]
  ├── source: str                    ├── source: str                    ├── source: str
  └── target: str                    ├── router: (state) -> str         ├── targets: [str]
                                     └── path_map: {str: str}           └── join: str?
```

边是值对象——它们不携带行为。执行器在 `_resolve_next()` 期间读取它们。

### 2.7 检查点器

```
Checkpointer (ABC)
  ├── put(key, state, parent_key, *, metadata=None)
  ├── get(key) -> Checkpoint
  └── list(thread_id) -> [key]
```

`Checkpointer` 抽象定义了 3 个操作。框架附带了用于开发的 `InMemoryCheckpointer`、用于单进程持久化的 `SqliteCheckpointer`，以及用于分布式部署的 `RedisCheckpointer`。

### 2.8 回调

```
Callback (Protocol)
  ├── on_graph_start/end
  ├── on_graph_error
  ├── on_node_start/end/error
  ├── on_state_update
  └── on_conditional_edge
```

每个方法都是可选的——`CallbackManager` 在分发前使用 `hasattr` 检查。这避免强迫用户实现空的存根方法。

### 2.9 暂停与恢复

节点可以通过抛出 `GraphExecutionPaused` 暂停执行：

```python
def human_input(state: State) -> dict:
    if not state.human_feedback:
        raise GraphExecutionPaused("等待人工输入")
    return {"result": process(state.human_feedback)}
```

暂停的执行可以从最后一个检查点恢复：

```python
result = compiled.resume("thread-1", state_type=MyState, updates={"human_feedback": "批准"})
```

---

## 3. 模块依赖图

```
                    ┌─────────────┐
                    │  __init__.py │  公共 API 表面
                    └──────┬──────┘
                           │
              ┌────────────┼──────────────────┐
              v            v                  v
        ┌──────────┐ ┌─────────┐ ┌──────────────────┐
        │ _types   │ │ _logging│ │ _visualize        │
        └──────────┘ └─────────┘ └──────────────────┘
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
        │_sqlite    │            └────┬─────┘
        │_redis     │                 │
        └──────────┘                 v
              │                 ┌──────────┐
              v                 │ _stream  │
        ┌──────────┐           └──────────┘
        │_callbacks│
        └──────────┘

内部模块（下划线前缀）：████████
公共模块：                    ████████
```

### 分层规则

1. **公共模块**（`state.py`, `pipeline.py`）可以被用户代码导入。
2. **内部模块**（`_*.py`）是实现细节——不保证导入稳定性。
3. **无循环导入**：依赖图是一个 DAG。会导致循环的跨模块引用使用方法体内部的惰性导入（例如，`_graph.py` 在 `CompiledGraph.invoke()` 内部导入 `SyncExecutor`）。
4. **`_types.py` 无依赖**（除了 `typing` 和 `typing_extensions`）。每个其他模块都依赖它。

---

## 4. 状态管理深入

### 4.1 合并管道

当节点返回 `StateUpdate` 字典时，执行器调用 `_apply(state, updates)`，它：

1. 检查状态是否有 `apply` 方法（Pydantic 路径）。
2. 调用 `state.apply(**updates)`。
3. `GraphState.apply()` 从 Pydantic 字段元数据懒加载构建 reducer 映射。
4. `merge_state()` 遍历更新键：
   - **无 reducer** 或 **OVERWRITE**：`resolved[key] = new_val`
   - **APPEND**：读取旧值，处理 `None`/列表/标量情况，产生扩展列表。
   - **REDUCE**：调用 `reducer(old_val, new_val)`。
5. 返回 `state.model_copy(update=resolved, deep=True)` —— 一个替换了已解析字段的深拷贝。

### 4.2 Reducer 缓存

reducer 映射（`field_name -> ReducerDescriptor`）每个状态类构建一次，作为 `ClassVar` 缓存。这避免了对每个 `apply()` 调用重新扫描 Pydantic 元数据。缓存在第一次 `apply()` 时懒加载填充——导入时没有急切自省。

### 4.3 Append 语义

`Append` 列表子类在运行时作为标记：

```python
# 在 merge_state 中：
if isinstance(new_val, Append):
    resolved[key] = old_val + list(new_val)   # 扩展
else:
    resolved[key] = [*old_val, *new_val]      # 连接
```

这允许节点返回普通列表进行批量扩展，或使用 `Append([single_item])` 进行增量添加，两者都触发相同的行为。

---

## 5. 图执行生命周期

### 5.1 同步执行 (`SyncExecutor.execute()`)

```
invoke(state)
  │
  ├── 构建配置 (recursion_limit, thread_id)
  ├── on_graph_start()
  │
  ├── [循环] 当 node != __end__:
  │     ├── 检查 recursion_limit
  │     ├── get_node(name)
  │     ├── 检查 fan-out（并行分支）
  │     │     └── _execute_fanout() ────► 合并所有分支状态
  │     ├── on_node_start()
  │     ├── node.invoke(state)  ────► 返回 StateUpdate
  │     │     └── [错误] on_node_error() + raise
  │     │     └── [暂停] GraphExecutionPaused → 保存检查点 + 返回
  │     ├── _apply(state, updates)  ──►  new_state
  │     ├── on_state_update()
  │     ├── on_node_end()
  │     ├── [检查点] put(key, new_state)
  │     ├── state = new_state
  │     ├── _resolve_next(graph, state)  ──► 下一个节点名
  │     │     ├── [条件边] router(state) ──►  path_map[key]
  │     │     ├── [扇出边] 目标列表
  │     │     └── [直连边] successors[0] 或 __end__
  │     └── [循环]
  │
  └── on_graph_end()
  return final_state
```

### 5.2 并行执行 (Fan-out)

当节点有扇出边时，`_execute_fanout()` 被调用：

- **同步模式**：每个目标分支按顺序运行到完成。
- **异步模式**：所有目标分支通过 `asyncio.gather()` 并发运行。
- 每个分支从相同的初始状态开始独立执行。
- 所有分支完成后，状态通过 `_merge_parallel_results()` 合并。
- 如果指定了 `join`，执行从连接节点继续；否则终止。

### 5.3 错误传播

节点错误会：
1. 通过 `logger.exception()` 以 ERROR 级别记录完整追踪。
2. 分发给 `Callback.on_node_error()`。
3. 重新抛出给调用者。

节点暂停通过 `GraphExecutionPaused` 异常处理，它保存检查点并返回当前状态（而不是传播错误）。

---

## 6. 流式架构

### 6.1 事件模型

```
StreamEvent
  ├── type: EventType
  ├── node: str        -- 节点名（图级别事件为空）
  ├── data: dict       -- 负载（状态快照、更新、错误）
  ├── step: int        -- 顺序步骤计数器
  ├── parent: str|None  -- 子图父节点
  └── metadata: dict
```

### 6.2 典型图的事件序列

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

---

## 7. 检查点模型

### 7.1 键结构

```
CheckpointKey = Tuple[thread_id: str, node_name: str, step_number: int]
```

检查点由 (thread, node, step) 唯一标识，形成追加日志。`parent_key` 字段将检查点链接到树形结构中，支持分支/合并模式。

### 7.2 检查点生命周期

1. 在每次节点调用前，当前状态被保存为以当前（即将执行的）节点名为键的检查点。
2. 检查点包含完整的状态快照（通过 `model_dump()`），因此图可以从此精确点恢复。
3. 暂停时，检查点元数据包含 `_resume_node` 字段，指示恢复时应重新运行哪个节点。

### 7.3 恢复流程

```python
# 暂停 → 保存检查点 → 返回状态
result = compiled.invoke(state)  # 在 human_input 节点暂停

# 用户提供输入并恢复
result = compiled.resume("session-1", state_type=MyState, updates={"input": "yes"})
# → 重新运行暂停的节点，使用更新后的状态
```

---

## 8. 对比

| 方面 | LangGraph | LangChain | GraphForge |
|---|---|---|---|
| **状态模型** | TypedDict + `__reducers__` 魔法 | BaseMessage / memory | Pydantic v2 + `node_field(merge=...)` |
| **类型安全** | 弱——TypedDict 无运行时验证 | 中等——部分泛型 | 强——`Generic[StateT]` 端到端 |
| **不可变性** | 原地修改状态 | 可变 | 不可变快照 |
| **合并策略** | 每个键的全局 reducer | 隐式 | 每个字段显式（overwrite/append/reduce） |
| **图组合** | 基于通道的内部机制 | 无（无图） | 子图和 Pipeline 作为一等公民节点 |
| **流式** | 不透明的通道事件 | Runnable events | 每步 `StreamEvent` 对象 |
| **异步支持** | 有异步 API | 有异步 API | 从第一天起原生同步+异步 |
| **检查点** | `BaseCheckpointSaver` | 无 | `Checkpointer` ABC + 内存/SQLite/Redis |
| **回调** | 无 | `BaseCallbackHandler` 层次结构 | 单个 `Callback` Protocol |
| **并行执行** | 基于通道 | 无 | `add_fanout()` + `asyncio.gather` |
| **依赖** | langchain-core, pydantic, 更多 | 许多 | Pydantic v2, typing_extensions |

---

## 9. 设计决策与理由

### 9.1 为什么用 Pydantic v2 而不是 TypedDict？

**决策**：状态模式 = Pydantic `BaseModel`。

**理由**：
- Pydantic 提供内置验证、序列化（`model_dump()`）和模式生成。
- TypedDict 是一个*静态*注解，没有运行时行为——LangGraph 必须使用自己的类级别的 `__reducers__` 字典和运行时注解解析来实现 reducer 机制，这很脆弱。
- `model_copy(update=..., deep=True)` 为我们免费提供了不可变快照。
- Pydantic 字段上的 `json_schema_extra` 是附加合并元数据的干净方式，无需弄乱元类。

### 9.2 为什么分离 Graph 和 CompiledGraph？

**决策**：两个类而不是一个带有 `compile()` 标志的类。

**理由**：
- 构建器（`Graph`）是可变的——你可以按任何顺序添加节点和边。
- 编译后的图（`CompiledGraph`）是不可变的——一旦冻结，拓扑永远不会改变。
- 这种分离允许编译后的图在编译时（而不是执行时）预先计算查找表（后继、条件）。
- 它反映了编译器的思维模型："构建时" vs "运行时"。

### 9.3 为什么将 Reducer 映射缓存为 ClassVar？

**决策**：`_reducers: ClassVar[Optional[_ReducerMap]] = None`，懒加载填充。

**理由**：
- 构建 reducer 映射需要遍历 `model_fields` 并检查 `json_schema_extra`——便宜但不免费。
- 每个类（而不是每个实例）做一次是安全的，因为字段元数据是类级别的。
- 懒加载填充避免了在模块导入时进行急切自省。
- ClassVar 范式也能正确处理子类。

### 9.4 为什么用 `from __future__ import annotations`？

**决策**：所有模块都使用 PEP 563 延后注解求值。

**理由**：
- 在大多数地方启用前向引用而无需字符串引号。
- 在模块级别避免循环导入问题——注解被惰性求值。
- 符合现代 Python 最佳实践。

---

## 10. 扩展点

### 10.1 自定义状态字段类型

通过子类化 `ReducerDescriptor` 并在 `merge_state()` 中处理新策略来添加新的合并策略。

### 10.2 自定义检查点器

实现 `Checkpointer` ABC：

```python
class PostgresCheckpointer(Checkpointer[StateT]):
    def put(self, key, state, parent_key=None, *, metadata=None): ...
    def get(self, key): ...
    def list(self, thread_id): ...
```

### 10.3 自定义日志

使用 Python 标准的 `logging` 配置：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# 或者：
from graphforge import configure_logging
configure_logging(level=logging.DEBUG)
```

---

## 术语表

| 术语 | 定义 |
|---|---|
| State | 代表流经图的累积数据的 Pydantic v2 BaseModel 子类 |
| StateUpdate | 节点返回的 `dict[str, Any]`，包含要合并到状态的字段 |
| Node | 包装了元数据（名称、类型）的可调用对象 `(state) -> StateUpdate` |
| Edge | 节点之间的连接——无条件（`DirectEdge`）或状态依赖（`ConditionalEdge`）或扇出（`FanOutEdge`） |
| Router | `ConditionalEdge` 用于选择下一个节点的可调用对象 `(state) -> str` |
| Graph | 有向执行图的可变构建器 |
| CompiledGraph | 准备好执行的不可变编译图 |
| Pipeline | 线性步骤序列（无分支，无路由） |
| Checkpoint | 特定 `(thread, node, step)` 处的状态快照 |
| Checkpointer | 用于存储/检索检查点的抽象接口 |
| Callback | 生命周期钩子的 Protocol |
| StreamEvent | 图执行期间发出的结构化事件 |
| MergeStrategy | 字段更新的折叠方式：`overwrite`, `append`, `reduce` |
| ReducerDescriptor | 指定合并策略的 Pydantic 字段元数据 |
| Append | 表示追加合并语义的列表子类标记 |
| Executor | 遍历图拓扑并调用节点的运行时组件 |
| GraphExecutionPaused | 节点用于暂停执行等待外部输入的异常 |
