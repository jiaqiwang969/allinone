# Qwen 语言网关与常驻服务设计

## 背景

当前 `allinone` 已经具备一条真实可跑的语言闭环：

- `run_runtime_observation` 会根据视觉结果构造 prompt
- `QwenClient` 会在服务器上调用离线 `Qwen3.5-9B`
- `language-smoke` 和 `run-research-step` 都已经可以真实走通

但这条链路有一个明确的工程瓶颈：

- 只要新起一个 Python 进程，9B 权重就会重新加载
- CLI 命令和未来线上 runtime 都天然是“多进程 / 多次调用”形态
- 结果就是系统大部分时间都消耗在“重新装载模型”，而不是“执行任务”

从第一性原理看，语言层真正要解决的不是“如何多打印一句话”，而是：

1. 把视觉状态翻译成稳定的结构化语言输出
2. 让这条翻译链路在 runtime 和 research 中都可重复复用
3. 让大模型的高装载成本只支付一次，而不是每次任务都支付一次

因此，单进程内缓存已经不够。下一步必须进入“跨命令、跨流程复用同一语言运行时”的阶段。

## 目标

在不破坏当前 DDD 分层的前提下，实现一条统一语言基础设施：

- 服务器上常驻一个 Qwen 推理服务
- CLI、runtime、research 都优先走这个服务
- 服务不可用时，仍保留本地直连或 mock 的回退路径
- 业务层不感知“本地模型”还是“远端服务”，只依赖文本生成边界

具体来说，本次设计要满足：

- `language-smoke` 不再因为新进程而重复加载 9B
- `run-research-step` 的每次实验批次优先复用常驻服务
- 后续接 M400 在线作业时，不需要重写语言调用方式

## 方案比较

### 方案 A：只扩大批处理进程复用

做法：

- 继续沿用本地 `QwenClient`
- 只在 `run-research-step` 或 `_CliExperimentBatchRunner` 中维持长生命周期对象

优点：

- 改动最小
- 风险最低

缺点：

- 只修研究闭环，不修真实 runtime
- 新命令进程仍会重新加载模型
- 无法形成后续线上统一基础设施

### 方案 B：直接做 Qwen 常驻服务

做法：

- 新增一个常驻 Qwen 服务进程
- CLI 与研究回放都通过 HTTP 或本机回环访问它

优点：

- 一次加载，多处复用
- 能直接解决跨进程重复装载问题

缺点：

- 如果服务接口直接暴露给业务入口，调用边界会比较散
- 本地开发、测试和服务器运行容易分成两套逻辑

### 方案 C：公共语言网关 + Qwen 常驻服务

做法：

- 在 `infrastructure/language/qwen` 内建立统一语言网关边界
- 网关支持两种底层实现：
  - `direct local runtime`
  - `remote qwen service`
- CLI 和 runtime 只依赖网关

优点：

- 架构最稳
- 服务器部署与本地开发可共用一套调用协议
- 以后可平滑接更大模型、量化模型或别的语言模型服务

缺点：

- 比方案 B 多一层适配
- 初始实现稍多一些

## 设计结论

采用方案 C。

原因不是“为了更漂亮的架构”，而是因为 `allinone` 的目标本来就不是临时实验脚本，而是：

- `runtime`
- `research`
- `future online service`

这三条链最终要共用一套语言基础设施。公共网关是把这三条链提前统一，而不是未来再返工。

## 架构设计

### 1. 业务边界保持不变

`application` 层继续只依赖“文本生成器”或“语言输出边界”。

本次不允许：

- 让 `application` 直接依赖 HTTP
- 让 `domain` 知道服务地址或端口
- 让 CLI 在业务逻辑里硬编码网络细节

也就是说，语言服务化只发生在 `infrastructure` 和 `interfaces`。

### 2. 新增语言网关

在 `infrastructure/language/qwen` 中增加统一网关对象，职责如下：

- 读取语言运行配置
- 判断优先走服务还是本地直连
- 发起请求并返回净化后的文本
- 屏蔽具体底层实现差异

建议结构：

- `client.py`
  本地离线 Qwen 运行时
- `service_client.py`
  调用常驻 Qwen 服务
- `gateway.py`
  根据配置选择 `service` 或 `local`
- `schemas.py`
  服务请求与响应的轻量结构

这样 `gateway` 是统一入口，`client` 和 `service_client` 是两种 adapter。

### 3. Qwen 常驻服务

新增一个轻量 HTTP 服务进程，职责只有三件事：

1. 启动时加载一次 `Qwen3.5-9B`
2. 接收 prompt 请求
3. 返回净化后的文本结果和少量元数据

服务接口尽量极简：

- `GET /health`
  返回服务是否就绪
- `POST /generate`
  输入：
  - `prompt`
  - `max_new_tokens`
  - `temperature`
  输出：
  - `text`
  - `model_id`
  - `mode=service`

本次不做：

- 多租户
- 鉴权
- 流式输出
- 请求队列与优先级调度

因为当前阶段的核心目标只是“跨进程复用同一 9B 运行时”。

### 4. CLI 调用路径

CLI 层增加一个明确的新入口：

- `serve-qwen`

用来在服务器上启动常驻服务。

现有以下命令不改业务语义，只改语言底层实现：

- `language-smoke`
- `runtime-observation`
- `run-research-step`

它们通过统一语言网关调用。

默认优先级：

1. 若配置了服务地址且健康检查通过，则走 `service`
2. 否则若本地模型存在，则走 `local`
3. 否则退回 `mock`

这样现有 CLI 行为仍然可用，但服务器端可获得更优的真实运行性能。

### 5. 配置设计

新增一个语言网关配置 recipe，例如：

- `configs/model_recipes/qwen_gateway.yaml`

字段建议包括：

- `mode: auto`
- `service_url: http://127.0.0.1:8001`
- `service_timeout_seconds: 30`
- `model_id`
- `runtime_path`
- `device`
- `max_new_tokens`
- `temperature`

`mode` 支持：

- `auto`
- `service`
- `local`

这样既支持服务器正式部署，也支持本地开发调试。

### 6. 运维脚本

在 `ops/remote` 中补两类脚本：

- 启动脚本
  - 启动 Qwen 服务
  - 记录 PID / 日志
- 健康检查脚本
  - 轮询 `/health`
  - 判断服务是否 ready

必要时增加：

- 停止脚本
- 重启脚本

但第一版只要求“可启动、可检查、可被 CLI 使用”。

## 数据流

### Runtime 路径

1. YOLO / V-JEPA 先生成观测状态
2. `run_runtime_observation` 构造 guidance prompt
3. 语言网关决定：
   - 调服务
   - 或调本地离线 Qwen
4. 返回净化后的结构化文本
5. 解析为 `operator_message / suggested_action / confidence / evidence_focus`

### Research 路径

1. `run-research-step` 读取 replay manifest
2. 每条样本都会调用 runtime 观察逻辑
3. runtime 通过语言网关请求语言输出
4. 常驻服务复用同一个已加载模型
5. `autoresearch` 再比较 candidate policy 的结果差异

这意味着语言层终于成为“可复用基础设施”，而不是“每次实验里反复初始化的重对象”。

## 失败与回退策略

必须明确处理三类失败：

### 1. 服务未启动

行为：

- 健康检查失败
- 网关自动切回 `local`
- 若本地模型也不可用，再退 `mock`

### 2. 服务请求超时

行为：

- 单次请求返回错误
- 网关可回退到 `local`
- CLI 输出仍保持结构化

### 3. 服务返回脏输出

行为：

- 仍复用 `QwenClient` 已有的输出净化逻辑
- 服务端和本地直连路径必须共用同一套文本清洗规则

这样能保证“运行方式不同，但语言结果格式一致”。

## 测试策略

至少覆盖四类验证：

### 1. 网关选择逻辑

- 服务可用时优先走服务
- 服务不可用时回退本地
- 本地不可用时回退 mock

### 2. 服务接口行为

- `health` 可返回 ready
- `generate` 能返回结构化文本
- 返回值会经过净化

### 3. CLI 行为

- `language-smoke` 可通过网关走服务
- `_CliRuntimeTextGenerator` 不需要知道具体底层实现

### 4. 远端真实验证

在 `dell@192.168.1.104` 上验证：

- 启动 Qwen 服务只加载一次权重
- 连续多次 `language-smoke` 不再每次重新装载 427 shards
- `run-research-step` 能用新服务完成一次真实实验

## 边界约束

本次明确不做：

- 把 YOLO、V-JEPA 也一起服务化
- 构建通用 GPU 调度平台
- 上消息队列
- 做复杂微服务拆分

因为目前最关键的是先把语言层从“重初始化脚本”变成“可复用基础设施”。

这一步做对之后，后面不管是统一在线 runtime，还是进一步把 `YOLO + V-JEPA + Qwen + autoresearch` 收束进更强的一体化模型，工程底座都会稳很多。
