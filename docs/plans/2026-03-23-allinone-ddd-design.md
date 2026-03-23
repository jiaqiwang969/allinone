# Allinone DDD 设计文档

## 1. 设计目标

`allinone` 的目标不是把 `YOLO / V-JEPA / Qwen / autoresearch` 机械拼接成一个大目录，而是围绕“远程工业作业会话闭环”重建一个统一产品仓。

这个系统要解决的核心问题不是单张图分类，而是一个完整作业闭环：

1. 识别当前任务和阶段
2. 判断镜头是否正确
3. 产生下一步动作建议
4. 驱动录制、抓图和证据生成
5. 判断证据是否满足验收要求
6. 将运行事实回流到研究闭环中持续优化

## 2. 第一性原理

系统中四类能力分别服务不同层面：

- `YOLO / Ultralytics` 负责空间定位问题，即“它在哪儿”
- `V-JEPA` 负责结构与时序表征问题，即“它处于什么状态、过程是否正常”
- `Qwen / LLM` 负责语言组织与解释问题，即“怎么把判断变成人能用的提示、任务卡和结构化输出”
- `autoresearch` 负责研究控制平面，即“怎么持续变好”

真正的 `all in one` 不是长期保留多套平行主系统，而是把这些能力蒸馏、压缩、固化进统一学生模型和统一业务事件链。

## 3. 领域划分

### 3.1 核心域

核心域定义为：`远程工业作业会话闭环`

其核心聚合根为：

- `WorkSession`
- `GuidanceLoop`
- `EvidenceBundle`
- `ExperimentRun`

其中：

- `WorkSession` 是业务执行面的聚合根
- `ExperimentRun` 是研究控制面的聚合根

### 3.2 子域

系统分为以下子域：

- `session`：作业会话、任务卡、阶段推进、操作者、设备
- `guidance`：取景引导、动作建议、开始录制/停止录制判定
- `evidence`：截图、录像、证据完整性、验收 readiness
- `perception`：ROI、目标定位、状态识别、结构理解
- `language`：提示语、解释、任务说明、结构化输出
- `research`：实验配方、评估、回放、keep/discard、策略提升
- `shared`：共享值对象、错误、时间、标识等横切对象

## 4. 聚合、实体和值对象

### 4.1 聚合

建议第一版保留四个主聚合：

- `WorkSession`
- `GuidanceLoop`
- `EvidenceBundle`
- `ExperimentRun`

### 4.2 关键实体

- `WorkSession`
- `StageExecution`
- `Clip`
- `FrameSample`
- `EvidenceItem`
- `GuidanceDecision`
- `PerceptionObservation`
- `LanguageResponse`
- `CandidateModel`
- `EvaluationReport`

### 4.3 关键值对象

- `BoundingBox`
- `CenterOffset`
- `FillRatio`
- `VisibilityScore`
- `ReadabilityScore`
- `StageType`
- `PromptAction`
- `EvidenceType`
- `EvidenceQuality`
- `ModelRecipe`
- `DataRecipe`
- `LossRecipe`
- `RuntimePolicy`
- `AcceptanceRule`

## 5. 领域不变量

第一版必须保持以下不变量：

1. 一个 `WorkSession` 必须有明确 `task_type`
2. 一个 `WorkSession` 任一时刻只能有一个 `current_stage`
3. 一个 `GuidanceDecision` 必须来源于某次 `PerceptionObservation`
4. `start_recording` 只能在 `ready` 条件成立后触发
5. `stop_recording` 必须和证据完成、阶段完成或失败退出相关
6. 一个 `EvidenceBundle` 只有满足 `AcceptanceRule` 后才能标记为 `acceptable`
7. 一个 `ExperimentRun` 只有通过 guardrail 才能进入 `kept`

## 6. 运行时命令流与事件流

### 6.1 主命令流

第一版在线业务命令建议为：

- `OpenWorkSession`
- `AttachTaskPlan`
- `StartStage`
- `IngestObservationWindow`
- `RequestGuidanceDecision`
- `StartRecording`
- `StopRecording`
- `CaptureEvidence`
- `AssessEvidenceBundle`
- `CompleteStage`
- `CloseWorkSession`

### 6.2 主事件流

建议沉淀以下核心事件：

- `WorkSessionOpened`
- `TaskPlanAttached`
- `StageStarted`
- `ObservationWindowIngested`
- `PerceptionObserved`
- `GuidanceDecisionMade`
- `RecordingStarted`
- `RecordingStopped`
- `EvidenceCaptured`
- `EvidenceBundleAssessed`
- `StageCompleted`
- `WorkSessionClosed`

异常事件至少包括：

- `GuidanceDecisionRejected`
- `EvidenceAssessmentFailed`
- `StageTransitionBlocked`
- `RuntimeFallbackTriggered`

### 6.3 业务闭环

第一版在线主闭环定义为：

`Observation -> Decision -> Action -> Evidence -> Re-assess`

### 6.4 研究闭环

研究平面命令建议为：

- `RegisterExperimentRecipe`
- `MaterializeDatasetView`
- `GeneratePseudoLabels`
- `TrainCandidateModel`
- `RunOfflineEvaluation`
- `RunRuntimeReplay`
- `JudgeCandidate`
- `PromoteCandidate`
- `PromoteRuntimePolicy`
- `ArchiveExperimentRun`

研究闭环本质为：

`出题 -> 配数据 -> 训练 -> 回放 -> 裁判 -> 保留/淘汰`

## 7. 分层架构

项目采用严格分层：

```text
src/allinone/
├── domain/
├── application/
├── infrastructure/
├── interfaces/
└── bootstrap/
```

依赖方向必须单向：

- `domain` 不依赖任何外层
- `application` 依赖 `domain`
- `infrastructure` 实现 `domain/application` 定义的 port
- `interfaces` 调用 `application`
- `bootstrap` 负责实际装配

## 8. Python 包与模块规范

### 8.1 领域包结构

每个子域统一采用如下结构：

```text
src/allinone/domain/<subdomain>/
├── entities.py
├── value_objects.py
├── commands.py
├── events.py
├── services.py
├── policies.py
├── repositories.py
└── errors.py
```

### 8.2 基础设施层定位

`YOLO / V-JEPA / Qwen / autoresearch` 都只能存在于 `infrastructure` 中，不得直接泄漏到领域层。

建议结构：

```text
src/allinone/infrastructure/
├── perception/
│   ├── yolo/
│   ├── vjepa/
│   ├── mediapipe/
│   └── fusion/
├── language/
│   └── qwen/
├── research/
│   └── autoresearch/
├── datasets/
├── persistence/
├── devices/
└── telemetry/
```

## 9. 现有资产的迁移定性

### 9.1 `autoresearch`

定性为最重要的逻辑来源，应拆入：

- `domain/research`
- `application/research`
- `infrastructure/research/autoresearch`
- `experiments`

### 9.2 `ultralytics`

定性为上游感知引擎，不应成为业务顶层目录。建议以受控副本或外部依赖形式保留，通过 `YOLO adapter` 接入。

### 9.3 `V-JEPA`

当前顶层更多是资料资产。文档类内容迁入 `docs`，真正编码接入时通过 `VJEPA adapter` 接入，不直接污染核心域。

### 9.4 `data`

属于数据资产，不进入 git。只迁移：

- schema
- manifest
- mapping 规则
- recipe
- 物化脚本

## 10. 第一阶段范围

第一阶段只围绕以下主闭环展开：

1. 头戴视频取景引导
2. 自动开始录制 / 停止录制
3. 有效证据抓图与归档 readiness
4. 基于 `autoresearch` 的 runtime replay 优化

以下内容不进入第一阶段主线：

- OCR / 仪表读数主线
- 深层缺陷诊断
- 多设备协同
- 全量端到端一体大模型训练

## 11. 迁移策略

迁移遵守五条规则：

1. 迁“能力”，不迁“历史形态”
2. 迁“主链路”，不迁“所有东西”
3. 上游框架不入侵核心域
4. 数据、权重、缓存不进 git
5. `31-allinone` 成为最终产品主仓，旧目录保留为历史资产来源

建议迁移顺序：

1. 先立新仓骨架
2. 先迁 `session / guidance / evidence / perception / research`
3. 再迁 `YOLO / V-JEPA / autoresearch` adapter
4. 再迁运行与部署入口
5. 最后接入 `Qwen3.5-9B` 语言层

## 12. 架构验收标准

判断 `allinone` 是否真正成型，至少看以下六条：

1. `31-allinone` 成为唯一主仓
2. 一条业务主链在新仓可跑通
3. `YOLO / V-JEPA / Qwen / autoresearch` 全部通过 adapter 接入
4. 在线事件流与研究事件流统一
5. 新功能默认进入新仓，不再继续堆在旧原型仓
6. 旧仓只做历史参考，不再承担主开发职责

## 13. 结论

`31-allinone` 不是“29-autoresearch 的整理版”，而是一个以“远程工业作业会话闭环”为核心的新产品仓。

这个仓库的本质是：

- 以 DDD 重建业务边界
- 以 adapter 挂载模型能力
- 以统一事件链连接业务闭环和研究闭环
- 以 `autoresearch` 持续优化运行策略和模型配方
