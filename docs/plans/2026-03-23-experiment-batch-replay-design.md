# 批量离线回放实验链设计

## 背景

`allinone` 当前已经打通了单条 clip 的运行主链：

1. `analyze-clip`
2. `build-observation-payload`
3. `runtime-observation`
4. `Qwen` 中文解释

这说明系统已经具备“看一段视频并给出动作提示”的单次推理能力。

但对 `autoresearch` 来说，单次推理还不够。真正需要的是一条可批量执行、可回放、可比较、可归档的实验链。否则：

- 没法稳定复现同一批视频上的结果
- 没法比较不同候选配置
- 没法形成后续训练/微调的监督资产
- 没法把 `YOLO / V-JEPA / guidance / Qwen` 串成真正的研究闭环

## 问题定义

我们现在缺的不是模型本身，而是“批量实验控制面”。

第一条正式实验链要解决的问题很具体：

- 给一批 clip 清单
- 按统一流程批量跑完整链路
- 为每个 clip 产出结构化结果
- 为整批 clip 产出 summary
- 把实验目录沉淀成后续 `autoresearch` 可直接消费的资产

这条链的目标不是马上训练大模型，而是先把“实验样本如何被稳定处理、记录、评估、比较”这件事做扎实。

## 目标

新增一个批处理入口，例如：

```bash
python3 -m allinone.interfaces.cli.main run-experiment \
  --manifest experiments/manifests/m400_phase1_demo.jsonl \
  --run-dir experiments/runs/run-2026-03-23-m400-phase1 \
  --candidate baseline \
  --yolo-model /home/dell/yolo11n.pt \
  --vjepa-repo /home/dell/vjepa2 \
  --vjepa-checkpoint /home/dell/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
```

它的职责是：

- 读取 manifest
- 对每个 clip 执行完整推理链
- 落地 `raw / payload / result / summary`
- 保留后续 `judge / replay / compare` 所需的最小研究元数据

## 为什么先做批量离线回放

有三个方向：

1. 先做批量离线回放
2. 先做单场景深度业务化
3. 直接做完整候选对比和自动选优

本轮选择 1，原因是：

- 最贴近 `autoresearch` 的核心价值
- 能最快形成“数据资产 + 结果资产 + 可复跑资产”
- 不会过早把系统绑死在某一个具体工业任务上
- 后面无论是做控制柜、仪表、卡尺，还是维修巡检，都可以共用这条实验底座

## 输入设计

第一阶段使用 `jsonl manifest`，每行一个 clip 样本。

建议字段：

- `clip_id`
- `clip_path`
- `target_labels`
- `task_type`
- `expected_action`
- `notes`

示例：

```json
{
  "clip_id": "cabinet-001",
  "clip_path": "/data/m400/cabinet-001.mp4",
  "target_labels": ["meter"],
  "task_type": "view_guidance",
  "expected_action": "left",
  "notes": "目标偏右，期望提示左移"
}
```

### 为什么只要求 `expected_action`

因为第一条实验链的目标不是做复杂标注平台，而是先回答一个更基本的问题：

“面对这段视频，系统最后给出的动作建议对不对？”

也就是说，先用 `expected_action` 这种粗标签建立最小可用评估闭环。后面再逐步增加：

- 目标是否完整进入框
- 是否到达可录制状态
- 哪一帧最适合作为证据
- 语言解释是否符合人工预期

## 运行链设计

每条 manifest 样本固定执行四段：

### 1. Clip 感知层

调用 `build_raw_perception_payload_from_clip(...)`，得到：

- `prediction_rows`
- `image_size`
- `best_frame_index`
- `visibility_score`
- `readable_ratio`
- `stability_score`
- `alignment_score`

### 2. 标准化层

调用 `build_observation_payload_from_raw(...)`，得到业务层稳定消费的标准 payload。

### 3. 运行时决策层

调用统一的 runtime 用例，得到：

- `guidance_action`
- `reason`
- `language_action`
- `confidence`
- `operator_message`
- `language_source`

这里不应该继续把逻辑藏在 CLI 内部 helper 中，而应抽成可复用的 application/runtime 用例。这样批处理和 CLI 才能共用同一条业务路径。

### 4. 结果归档层

将 manifest 输入、clip 感知结果、标准 payload、运行时决策、评估字段合并成一个 `result row`。

## 运行目录设计

建议每次实验落在一个独立目录下：

```text
experiments/runs/run-2026-03-23-m400-phase1/
├── manifest.jsonl
├── results.jsonl
├── summary.json
├── raw/
├── payload/
└── logs/
```

目录职责：

- `manifest.jsonl`
  - 固化本次实验输入
- `raw/`
  - 每条 clip 的原始感知输出
- `payload/`
  - 标准 observation payload
- `results.jsonl`
  - 每条 clip 的最终结构化结果
- `summary.json`
  - 全批次统计指标
- `logs/`
  - 失败样本、运行时间、异常信息

这个目录结构本质上就是第一版研究数据平面。后面 `autoresearch` 可以直接消费，不需要再重新设计输入输出格式。

## 结果行设计

建议每条 `results.jsonl` 至少包含：

- `clip_id`
- `candidate_name`
- `task_type`
- `target_labels`
- `expected_action`
- `guidance_action`
- `language_action`
- `action_match`
- `target_detected`
- `best_frame_index`
- `visibility_score`
- `readable_ratio`
- `stability_score`
- `alignment_score`
- `operator_message`
- `language_source`
- `error`

这样后面能直接回答：

- 哪些 clip 判断对了
- 哪些 clip 根本没看到目标
- 哪些 clip 是 `V-JEPA` 质量太差
- 哪些 clip 是 guidance 策略问题
- 哪些 clip 是语言解释跑偏

## 第一阶段评估指标

第一阶段只做三类核心指标：

### 1. `action_match_rate`

`guidance_action == expected_action` 的比例。

这是最基础的业务指标，回答“这条链最后提示方向对不对”。

### 2. `target_detected_rate`

`prediction_rows` 非空的比例。

这是感知入口健康度指标，回答“系统至少看到了目标没有”。

### 3. `usable_clip_rate`

满足最小质量阈值的比例，例如：

- `visibility_score >= 0.5`
- `readable_ratio >= 0.5`

这是视频质量可用性指标，回答“输入视频本身适不适合被系统消费”。

第一阶段不急着做复杂 judge，因为先把这三个指标稳定下来，已经足够支持第一轮策略比较。

## 与 autoresearch 的连接方式

本轮不是完整的 `candidate compare` 系统，但必须保留它的接口位。

因此每次运行都保留：

- `candidate_name`
- `run_dir`
- `summary.json`
- `results.jsonl`

后续 `autoresearch` 的角色是：

1. 读取多个 `run_dir`
2. 对比不同 `candidate_name`
3. 基于 `summary.json` 和 `results.jsonl` 做打分
4. 决定下一轮改什么参数、提示词、阈值或模型组合

也就是说，本轮做的是“研究底座”，不是“自动选优终局”。

## 架构落点

建议职责划分如下：

- `application.runtime`
  - 增加一个可复用的 runtime 结果用例，负责组合 guidance 与语言解释
- `application.research`
  - 增加批量实验运行用例
- `infrastructure.research.autoresearch`
  - 负责 run 目录写入、summary 统计与研究输出适配
- `interfaces.cli`
  - 新增 `run-experiment`

这样可以保持 DDD 边界清晰：

- `domain` 不知道 YOLO/V-JEPA/Qwen
- `application` 只负责编排
- `infrastructure` 负责具体持久化和研究输出格式
- `CLI` 只是入口

## 非目标

本轮明确不做：

- 多 candidate 自动并行比较
- 自动 judge 选优
- 完整会话级 evidence 沉淀
- 在线实时流式推理
- 新的训练或微调流程
- 大规模标注管理

## 成功标准

本轮完成后，需要满足：

1. 能通过一个 manifest 批量跑完整 clip 实验
2. 每条 clip 都能生成 `raw / payload / runtime result`
3. 每批实验都能生成 `results.jsonl + summary.json`
4. `summary` 至少包含 `action_match_rate / target_detected_rate / usable_clip_rate`
5. 同一批 clip 可以被重复回放，结果目录结构稳定
6. 服务器上能对真实或半真实 clip 批量执行一次完整 run
