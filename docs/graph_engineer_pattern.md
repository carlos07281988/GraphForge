 # Graph Engineer 模式 — 新一代 AI 智能体架构
 
 > **最新更新**：2026-07-22
 > 
 > 本文档总结 "Graph Engineer"（图工程师）模式——AI 智能体架构的最新一代范式，
 > 并阐述 GraphForge 作为该模式原生实现的设计理念与能力。
 
 ---
 
 ## 1. 什么是 Graph Engineer 模式
 
 **Graph Engineer** 是一种以**有向状态图**为计算原语的 AI 智能体编排范式。
 它将智能体工作流建模为图，其中节点是操作（LLM 调用、工具执行、人工审批、子图编排），
 边携带条件路由、并行扇出和状态合约。这取代了前两代模式：
 
 | 世代 | 模式 | 代表框架 | 心智模型 |
 |------|------|----------|----------|
 | 1.0 | 链式（Chain） | LangChain | 线性管道 |
 | 2.0 | 循环（ReAct） | 原生 OpenAI Function Calling | 思考→行动→循环 |
 | 3.0 | **图工程师（Graph Engineer）** | LangGraph, **GraphForge** | 有状态执行图 |
 
 ### 核心洞察
 
 智能体工作流天然是图结构：LLM 调用可能触发工具执行，工具结果可能改变路由决策，
 子任务可以并行执行，人工审批可能在任何节点介入。线性链和简单循环无法自然表达
 这种拓扑——图是匹配问题结构的抽象层级。
 
 ---
 
 ## 2. 关键参考文献
 
 下列资源定义了 Graph Engineer 模式的理论和实践基础，GraphForge 在其设计中
 直接引用或对标了这些工作：
 
 | 资源 | 链接 | 与本模式的关系 |
 |------|------|----------------|
 | **LangGraph 文档** | https://langchain-ai.github.io/langgraph/ | 开创性的图智能体框架，首次将状态图引入 LLM 工作流 |
 | **Anthropic: Building Effective Agents** | https://www.anthropic.com/engineering/building-effective-agents | 将图工作流作为生产级智能体设计的推荐模式 |
 | **Google A2A 协议** | https://google.github.io/A2A/ | 智能体间通信标准，使图可以跨框架通信 |
 | **MCP (Model Context Protocol)** | https://modelcontextprotocol.io/ | 工具/资源的标准化协议，被 GraphForge 集成 |
 | **OpenAI Structured Outputs** | https://platform.openai.com/docs/guides/structured-outputs | 类型化节点输出，与 Pydantic 模型结合使用 |
 | **Pydantic v2** | https://docs.pydantic.dev/latest/ | GraphForge 的状态模型基础 |
 
 GraphForge 的架构文档（[architecture.zh-CN.md](architecture.zh-CN.md)）提供了
 与 LangGraph 和 LangChain 的详细对比分析。
 
 ---
 
 ## 3. 模式设计原则
 
 ### 3.1 图即 Schema
 
 图的拓扑结构就是工作流的合约。边定义了哪些路径是合法的，路由函数编码了业务逻辑，
 状态类型约束了节点间的数据流。一个编译后的图是不可变的——执行期间不会再改变拓扑。
 
 ### 3.2 状态显式化
 
 每个字段必须声明其合并语义：覆盖（overwrite）、追加（append）还是归约（reduce）。
 不存在隐式行为。这保证了节点之间的合约是自文档化的。
 
 ### 3.3 不可变快照
 
 图中的每步执行都产生一个全新的状态快照。这使得：
 - **确定性**：相同输入 + 相同检查点 = 相同执行路径
 - **可调试**：任何步骤的状态都可事后检视
 - **安全性**：节点不能意外地在步骤间泄漏副作用
 
 ### 3.4 可组合性
 
 编译后的图可以作为另一个图的节点。子图和 Pipeline 是一等公民。
 这使得复杂的编排可以从简单的组件层次化构建。
 
 ### 3.5 透明执行
 
 所有执行路径（同步、异步、流式）发射结构化事件。没有隐藏的内部通道或协议。
 执行可以被录制、回放、可视化。
 
 ---
 
 ## 4. GraphForge 的架构对照
 
 GraphForge 是 Graph Engineer 模式的一个从头构建的、类型安全的实现。
 以下从模式角度分析其架构模块：
 
 ### 4.1 状态图基础
 
 | 模式要求 | GraphForge 实现 | 文件 |
 |----------|----------------|------|
 | 有状态节点 | `Graph.add_node(name, fn)` — 节点是 `(State) -> dict` | `_graph.py` |
 | 条件路由 | `Graph.add_conditional_edges(src, router, path_map)` | `_graph.py`, `_edge.py` |
 | 无条件边 | `Graph.add_edge(src, target)` | `_graph.py`, `_edge.py` |
 | 并行扇出 | `Graph.add_fanout(src, targets, join=name)` | `_graph.py`, `_edge.py` |
 | 终止信号 | `"__end__"` 哨兵 | `_types.py` |
 | 图编译 | `Graph.compile() -> CompiledGraph`（不可变） | `_graph.py` |
 | 执行器 | `SyncExecutor` / `AsyncExecutor` | `_executor.py` |
 | 流式 | `CompiledGraph.stream()` / `astream()` | `_stream.py`, `_executor.py` |
 
 ### 4.2 状态管理
 
 | 模式要求 | GraphForge 实现 | 文件 |
 |----------|----------------|------|
 | 类型化状态 | `GraphState(BaseModel)` — Pydantic v2 | `state.py` |
 | 字段级合并 | `node_field(merge="append" / "overwrite" / "reduce")` | `state.py` |
 | 不可变性 | `state.apply(**updates)` — 返回新实例 | `state.py` |
 | 显式追加 | `Append([item])` 标记 | `state.py` |
 | 可配置字段 | `node_field(configurable=True)` | `state.py` |
 | 冲突策略 | `On.REPLACE / APPEND / IGNORE / ERROR` | `_edge.py` |
 
 ### 4.3 智能体编排模式
 
 | 模式 | GraphForge 工厂函数 | 拓扑结构 |
 |------|---------------------|----------|
 | ReAct | `create_react_agent()` | Agent → Tools → Agent → ... → End |
 | 监督者/工人 | `create_supervisor_worker()` | 监督者 → 工人 → 监督者 → ... → 完成 |
 | 群（Swarm） | `create_swarm()` | Router → AgentA → Router → AgentB → ... → End |
 | 委托 | `create_delegation_agent()` | 编排者 → 子图 → 编排者 |
 | 人工审批 | `ApprovalNode(fn)` | 包装任意节点，执行前审批 |
 
 所有模式实现在 [`agents/patterns.py`](../graphforge/agents/patterns.py) 中。
 
 ### 4.4 节点类型体系
 
 | 节点类型 | 用途 | GraphForge 实现 |
 |----------|------|----------------|
 | 函数节点 | 纯计算 | `def fn(state) -> dict` |
 | 异步节点 | I/O 绑定操作 | `async def fn(state) -> dict` |
 | 生成器节点 | Token 级流式 | `def fn(state): yield update` |
 | 子图节点 | 图组合 | `CompiledGraph` as node |
 | Pipeline 节点 | 线性序列 | `Pipeline(steps)` as node |
 | 工具节点 | LLM + 工具调用 | `ToolNode(llm_func, tools)` |
 | 检索节点 | RAG 知识检索 | `RetrievalNode(vectorstore)` |
 | 图像节点 | 多模态输入/输出 | `ImageNode()` |
 
 ### 4.5 企业级能力
 
 | 能力 | GraphForge 模块 | 描述 |
 |------|----------------|------|
 | 检查点 | `_checkpoint*.py` | SQLite / Redis / Postgres 后端 |
 | 持久化存储 | `store.py`, `store_redis.py` | 跨线程 KV 存储 |
 | 回调系统 | `_callbacks.py` | 钩子：graph_start/end, node_start/end, state_update |
 | 中间件 | `_middleware.py` | 状态转换的前/后处理管道 |
 | 守卫 | `guardrails.py` | `InputGuardian` / `OutputGuardian` |
 | 评估 | `eval.py` | `evaluate(graph, cases)` — 按图评测 |
 | 成本追踪 | `_callbacks.py` CostCallback | 每节点 Token + 成本归因 |
 | 优化器 | `_optimizer.py` AutoOptimizer | 自动检测独立路径并并行化 |
 | 时间线 | `_timeline.py` TimelineRecorder | 完整的状态转换录制与回放 |
 | 可视化 | `_mermaid.py`, `_visualize.py` | Mermaid / DOT 图导出 |
 | 自动 Web UI | `_dashboard.py` | 任意 CompiledGraph 生成交互式仪表盘 |
 | 分布式执行 | `distributed.py` | ThreadPool / ProcessPool worker |
 
 ### 4.6 智能体间通信
 
 | 协议 | GraphForge 模块 | 能力 |
 |------|----------------|------|
 | A2A（Agent-to-Agent） | `a2a/` | 与任意 A2A 兼容的智能体框架互操作 |
 | MCP（Model Context Protocol） | `mcp/` | 工具/资源的标准化暴露和调用 |
 | REST / SSE | `_http_server.py` | HTTP 调用和流式响应 |
 | WebSocket | `_http_server.py` /ws | 双向实时通信 |
 | Swagger | `serve.py` | 自动生成 OpenAPI 文档 |
 | Agent Card | `a2a/_server.py` | `.well-known/agent-card` 发现端点 |
 
 ### 4.7 一键部署
 
 ```python
 from graphforge.serve import serve
 
 serve(compiled_graph, host="0.0.0.0", port=8080)
 # 同时提供：
 #   POST /invoke          — 同步调用
 #   POST /stream          — SSE 流式
 #   GET  /ws              — WebSocket
 #   POST /mcp/call        — MCP 调用
 #   POST /tasks/send      — A2A 任务
 #   GET  /.well-known/*   — 发现端点
 #   GET  /docs            — Swagger UI
 #   GET  /dashboard       — 图可视化仪表盘
 ```
 
 ---
 
 ## 5. 与 LangGraph 的对比
 
 GraphForge 在设计上对标 LangGraph，但在关键维度上选择了不同的
 工程取舍：
 
 | 维度 | LangGraph | GraphForge |
 |------|-----------|------------|
 | **状态模型** | TypedDict + 隐式 `__reducers__` | Pydantic v2 + 显式 `node_field(merge=...)` |
 | **类型安全** | 弱（TypedDict 运行时不稳定） | 强（`Generic[StateT]` 贯穿全栈） |
 | **不可变性** | 原地修改 | 每步 `model_copy()` 新快照 |
 | **合并语义** | 每键全局 reducer | 每字段显式声明（覆盖/追加/归约） |
 | **图组合** | 基于通道的消息传递 | 子图 + Pipeline 作为一等节点 |
 | **流式事件** | 不透明通道事件 | 结构化 `StreamEvent` 对象 |
 | **API 表面积** | 数百个类 | ~20 个公开导出 |
 | **依赖** | langchain-core, pydantic 等 | Pydantic v2 + typing_extensions |
 
 ---
 
 ## 6. GraphForge 的独特创新
 
 以下功能在现有图智能体框架（包括 LangGraph）中不具备，是 GraphForge 的特色：
 
 ### 6.1 自动图并行化（AutoOptimizer）
 
 `AutoOptimizer` 静态分析节点的状态字段读写集，检测不存在依赖关系的独立路径，
 自动插入扇出边。无需用户标注。
 
 ### 6.2 执行时间线录制与回放（TimelineRecorder）
 
 全量录制每一步的完整状态转换（节点名、步骤号、状态前后快照、更新详情、耗时），
 支持事后调试和回放分析。
 
 ### 6.3 统一 serve() — 五合一协议
 
 一条命令同时启动 REST、SSE、WebSocket、MCP、A2A 五种协议，
 加上自动 Swagger 文档和图仪表盘。
 
 ### 6.4 显式冲突策略
 
 并行分支更新同一字段时，可配置策略：REPLACE（最后写入者获胜）、
 APPEND（列表拼接）、IGNORE（首选写入者获胜）、ERROR（冲突时报错）。
 
 ### 6.5 Agent Evaluation 框架
 
 `evaluate(graph, cases)` 按图进行评测，而非按链。支持 exact_match、
 contains、json_match 和自定义评测指标。
 
 ---
 
 ## 7. 何时使用 Graph Engineer 模式
 
 适合：
 - 工作流涉及多个 LLM 调用和工具执行的组合
 - 执行路径有条件分支或循环
 - 需要并行执行子任务
 - 人工审批需要介入任意节点
 - 需要一个持久化的执行检查点用于恢复
 
 不适合：
 - 只需要一个简单的 LLM 调用 → 返回结果
 - 工作流完全可以由单次 LLM 调用完成
 - 没有条件分支、循环或并行的线性路径
 
 ---
 
 ## 8. 参考实现
 
 GraphForge 的完整实现和示例：
 
 - **[核心图引擎](..//graphforge/_graph.py)** — Graph / CompiledGraph
 - **[执行器](..//graphforge/_executor.py)** — 同步/异步/流式执行
 - **[智能体模式](..//graphforge/agents/patterns.py)** — 监督者/群/委托
 - **[ReAct 智能体](..//graphforge/agents/_react.py)** — 思考-行动循环
 - **[ToolNode](..//graphforge/agents/_tool_node.py)** — LLM 工具调用节点
 - **[A2A 协议集成](..//graphforge/a2a/)** — 智能体间通信
 - **[MCP 集成](..//graphforge/mcp/)** — 模型上下文协议
 - **[RAG 模块](..//graphforge/rag/)** — 知识检索节点
 - **[自动优化器](..//graphforge/_optimizer.py)** — 图并行化
 - **[时间线录制](..//graphforge/_timeline.py)** — 执行录制/回放
 - **[统一部署](..//graphforge/serve.py)** — 一键服务
 - **[示例](..//examples/)** — 完整使用案例
 - **[架构文档](architecture.zh-CN.md)** — 设计原理与对比
