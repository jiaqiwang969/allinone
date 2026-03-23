# Sensitive Policy Replay 设计

## 背景

当前 `run-research-step` 已经能完成：

- 生成 candidate policy
- 执行 replay
- judge 多个 candidate
- 选出最优 policy

但真实 demo 已经暴露出两个关键问题：

1. 只靠真实 clip 回放，candidate 很容易打平  
   因为 policy 变化只发生在 runtime guidance 阈值，而很多 clip 的观测值离阈值边界太远，所有 candidate 都会输出同样的动作。

2. 当前 judge 只能稳定观察 action 差异，观察不到 reason 差异  
   例如 `tighten_center_window` 在很多边界样本上会改变 `reason`，但不会改变 `action`。如果 judge 只看 `action_match_rate`，这个 candidate 就天然不可观测。

从第一性原理看，研究闭环要成立，必须同时满足：

1. `candidate` 能真实改变行为
2. 数据能把这种行为差异激发出来
3. 评估器能看见这种差异

现在第 1 点已经基本具备，第 2、3 点还没有补齐。

## 目标

补齐第一版“可观测、可复现、低成本”的敏感 replay 体系：

- `run-experiment` / `run-research-step` 不仅支持 `clip_path`，也支持冻结后的 `raw_payload_path`
- 用真实 clip 先跑出一个 base raw payload，再生成多个阈值边界 raw payload case
- manifest 可以同时标注：
  - `expected_action`
  - `expected_reason`
- judge 在已有 action 指标基础上，增加可选 `reason_match_rate`

这样一来：

- runtime policy 搜索不需要每轮都重跑 `YOLO + V-JEPA`
- `candidate-1 / candidate-2 / candidate-3` 都能被样本和 judge 真实区分
- 研究闭环第一次真正拥有“敏感数据面 + 可观测评估面”

## 方案比较

### 方案 A：只生成更多真实视频变体

- 基于真实 clip 做平移、裁剪、缩放
- 每次都重新跑 `YOLO + V-JEPA`

优点：

- 最接近真实视觉链路

缺点：

- 成本高
- 不稳定，YOLO 可能因为变形或黑边直接丢检
- 只解决“数据面”，不解决“reason 不可观测”

这个方案单独做不够。

### 方案 B：只手工写 payload 样本

- 直接构造 `prediction_rows / visibility_score / readable_ratio`
- 不再依赖真实视频

优点：

- 最快
- 最可控

缺点：

- 跟真实视频来源断裂
- 不利于后面形成“真实 clip -> 冻结 trace -> 研究 replay”主干

这个方案过于简化。

### 方案 C：真实 clip 冻结 trace + 边界 payload 生成 + reason-aware judge

- 先从真实 clip 导出 `raw_payload`
- 再基于这个 `raw_payload` 生成多个阈值边界 case
- manifest 支持 `raw_payload_path`
- judge 增加可选 `reason_match_rate`

优点：

- 保留真实视觉来源
- replay 成本低
- 样本可控
- 能观测到 action 和 reason 两类差异

缺点：

- 需要同时改 replay 输入和 judge 评分

这是推荐方案。

## 设计结论

采用方案 C。

第一版敏感 replay 系统由四部分组成：

1. `raw payload replay`
2. `boundary case generator`
3. `reason-aware result row`
4. `reason-aware judge`

## 架构落点

### `application.research.run_experiment_batch`

保持“回放编排”职责，但扩展输入模式。

每个 manifest row 允许两种来源：

- `clip_path`
- `raw_payload_path`

编排规则：

- 如果有 `raw_payload_path`，优先读取 raw payload 并直接进入 `build_observation_payload_from_raw`
- 否则继续走现有 `clip_path -> clip_analyzer`

这保证：

- 视频 replay 和冻结 payload replay 共用同一条 research 主链
- 不破坏 DDD 分层

### `infrastructure.research.autoresearch`

新增两个基础设施组件：

1. `RawPayloadLoader`
   - 从 JSON 读取冻结后的 raw perception payload

2. `GuidanceBoundaryDatasetBuilder`
   - 输入 base raw payload
   - 基于 bbox 几何关系生成边界样本
   - 写出：
     - `raw/*.json`
     - `manifest.jsonl`

### `run result / summary / judge`

结果行增加：

- `expected_reason`
- `guidance_reason`
- `reason_match`

summary 增加：

- `reason_match_rate`

judge 规则：

- 如果某个 run 的结果中存在 `expected_reason`
  - 则将 `reason_match_rate` 纳入总分
- 如果不存在
  - 则保持现有评分逻辑

这样可以保证向后兼容。

## 边界数据本体

第一版只围绕当前三种 mutation 生成样本：

### 1. `tight_center_boundary`

目标：

- `dx` 落在 `0.072 ~ 0.09`
- baseline 会给出：
  - `action=hold_still`
  - `reason=fully_centered`
- `tighten_center_window` 会给出：
  - `action=hold_still`
  - `reason=stabilize_before_capture`

这个样本主要区分 `reason`。

### 2. `direction_trigger_boundary`

目标：

- `dx` 落在 `0.153 ~ 0.18`
- baseline 会给出：
  - `action=hold_still`
- `earlier_direction_trigger` 会给出：
  - `action=left/right`

这个样本主要区分 `action`。

### 3. `oversize_boundary`

目标：

- `fill_ratio` 落在 `0.85 ~ 0.8925`
- baseline 会给出：
  - `action=backward`
- `allow_larger_target_before_backward` 会给出：
  - `action=hold_still`

这个样本也主要区分 `action`。

## 数据流

### 真实 clip 到 replay dataset

1. 先对真实 clip 执行一次 `analyze-clip`
2. 得到 base `raw_payload`
3. 用 `GuidanceBoundaryDatasetBuilder` 基于这个 base payload 生成边界样本
4. 自动写出 manifest
5. 用 `run-research-step` 在这些 replay case 上比较 candidate

### 研究闭环

1. 读取 base policy
2. 生成 candidate policies
3. 对同一份边界 manifest 回放
4. 记录：
   - `action_match_rate`
   - `reason_match_rate`
   - `target_detected_rate`
   - `usable_clip_rate`
5. judge 选优

## CLI 设计

新增命令：

`build-guidance-replay-dataset`

建议参数：

- `--input-raw`
- `--output-dir`
- `--target-label`

输出：

- `raw/tight_center_boundary.json`
- `raw/direction_trigger_boundary.json`
- `raw/oversize_boundary.json`
- `manifest.jsonl`

## 非目标

本轮不做：

- 多目标、多 bbox 联动 replay
- V-JEPA 信号自动扰动生成
- reason 的 LLM 语义判分
- 在线自动生成 replay dataset
- 多轮 frontier 数据库

## 成功标准

本轮完成后，应满足：

1. `run-experiment` 和 `run-research-step` 能回放 `raw_payload_path`
2. 系统能从真实 raw payload 自动生成边界 replay 数据
3. `reason_match_rate` 能进入 summary 和 judge
4. 真实服务器上能跑出一轮“不打平”的 research step
5. 生成的 replay dataset 可以作为后续 runtime policy 搜索的固定基准集
