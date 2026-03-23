# V-JEPA Clip 分析闭环设计

## 背景

当前 `allinone` 已经具备两条稳定主链：

1. `detect-image -> raw perception -> build-observation-payload -> runtime-observation`
2. `runtime-observation -> guidance -> Qwen 中文指令`

其中：

- `YOLO` 已经真实进入系统，负责单帧目标定位
- `V-JEPA` 还停留在 adapter 占位阶段，只能消费外部给定的质量分

这意味着系统已经会“看一张图里目标在哪”，但还不会“判断一段取景动作是否像一个合格证据片段”。

## 问题定义

对于 M400 头戴质检，单张图不够。

真正有价值的是一小段视频过程里的时序判断，例如：

- 镜头是否稳定
- 视角是否持续对准任务目标
- 操作员是否已经把目标推进到“可录制/可截图”状态
- 哪一帧最适合作为证据帧

这类问题不该继续交给 `YOLO`，而应由 `V-JEPA` 负责。

## 角色分工

### YOLO 的职责

- 找到目标是否出现
- 给出目标框、类别和置信度
- 在采样帧中选出几何上最优的候选证据帧

### V-JEPA 的职责

- 对短视频片段进行时序质量评估
- 判断镜头是否稳定、取景是否对齐、画面是否接近任务要求
- 输出 clip 级质量信号，而不是检测框

### Fusion 的职责

- 把 `YOLO` 的几何观测和 `V-JEPA` 的时序质量组合成统一的 `raw perception`
- 继续喂给现有 `build-observation-payload`

## 为什么选择 clip，而不是单图

有三个潜在方向：

1. 单图质量分
2. clip 时序质量分
3. 直接把 `V-JEPA` 做成第二个检测器

本轮选择 2，理由：

- `V-JEPA` 天然适合时序感知
- 头戴取景指导本质是动作过程评估，不是静态图分类
- 单图会把 `V-JEPA` 降级成普通图像质量打分器
- 把 `V-JEPA` 当检测器会破坏 `YOLO` 的清晰职责边界

## 目标形态

新增一个 clip 级入口：

```bash
python3 -m allinone.interfaces.cli.main analyze-clip \
  --clip <clip.mp4> \
  --yolo-model <yolo.pt> \
  --vjepa-repo <repo> \
  --vjepa-checkpoint <ckpt> \
  --targets meter \
  --output <raw.json>
```

其职责是分析一段短视频，输出统一的 `raw perception JSON`。

## 数据流

### 1. Clip Sampler

把一段短视频切成固定数量的采样帧，例如 8 帧或 16 帧。

第一阶段不做复杂自适应采样，只要求：

- 采样稳定
- 能回放
- 能被测试

### 2. YOLO Branch

对采样帧逐帧检测：

- 找到目标
- 计算每一帧的候选框
- 选出“目标最大、最居中、最像有效证据”的 `best_frame`

输出：

- `prediction_rows`
- `image_size`
- `target_labels`
- `best_frame_index`

### 3. V-JEPA Branch

对 clip 或采样帧序列运行 `V-JEPA`：

- 提取时序特征
- 计算 clip 级质量信号

第一阶段先输出：

- `visibility_score`
- `readable_ratio`
- `stability_score`
- `alignment_score`

其中前两个继续对接现有业务链，后两个先保留在 `raw perception` 中。

### 4. Raw Perception Fusion

建议输出结构：

```json
{
  "detections": {
    "prediction_rows": [...],
    "image_size": [810, 1080],
    "target_labels": ["meter"],
    "best_frame_index": 5
  },
  "vjepa": {
    "visibility_score": 0.86,
    "readable_ratio": 0.79,
    "stability_score": 0.91,
    "alignment_score": 0.84
  }
}
```

## 对现有主链的兼容策略

当前 `build-observation-payload` 已稳定消费：

- `prediction_rows`
- `image_size`
- `target_labels`
- `visibility_score`
- `readable_ratio`

因此本轮不修改业务层领域对象，只做向前兼容：

- `visibility_score/readable_ratio` 进入现有主链
- `stability_score/alignment_score` 先保留在 raw layer

这样不会打断现在已经能跑的 guidance/Qwen 链路。

## 与 autoresearch 的关系

这一轮不是训练大模型本体，而是让系统级闭环成立。

`autoresearch` 后续优化的对象会是：

- clip 采样帧数
- best-frame 选择策略
- V-JEPA 评分头参数或阈值
- guidance 阈值
- Qwen 提示词

闭环形式是：

`任务 clip 集 -> candidate 配置 -> analyze-clip -> runtime-observation -> 自动打分 -> 选优`

## 现有资产与复用原则

服务器已有可复用资产：

- V-JEPA 仓库：`/home/dell/vjepa2`
- checkpoint：`/home/dell/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`
- 历史适配样例：`/home/dell/autoresearch-m400-guidance/experiments/m400_guidance/vjepa_adapter.py`

本轮优先复用这些资产，不重建新的 V-JEPA 环境。

## 第一阶段非目标

本轮明确不做：

- 视频级训练/微调
- `V-JEPA` 新分类头训练
- 会话级多 clip 聚合
- evidence 自动沉淀
- 把 `stability_score/alignment_score` 直接改进领域对象

## 成功标准

本轮完成后，需要满足：

1. `analyze-clip` 能输出稳定的 `raw perception JSON`
2. `V-JEPA` 真实参与 clip 分析，而不是外部手工填分
3. 输出结果可继续走 `build-observation-payload -> runtime-observation`
4. 同一批 clip 可以用于后续 autoresearch replay 和候选比较
