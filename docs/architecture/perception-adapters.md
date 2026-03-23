# Perception Adapters

## 目标

把 `YOLO`、`V-JEPA` 和后续视觉模型统一收口到 `PerceptionObservation`，避免业务层直接依赖上游框架。

## 适配原则

- `YOLO`
  - 负责看见目标在哪里
  - 输出目标框、目标类别、置信度、可读区域候选
- `V-JEPA`
  - 负责理解结构是否稳定、视角是否合理、画面是否接近任务需要的状态
  - 输出结构表征、稳定性评分、可读性先验、动作前后时序特征
- `Fusion`
  - 把 `YOLO` 的几何结果和 `V-JEPA` 的语义结果融合
  - 产出领域对象 `PerceptionObservation`

## 对业务层暴露的统一对象

`PerceptionObservation` 只保留业务真正关心的字段：

- `visibility_score`
- `readable_ratio`
- `fill_ratio`
- `center_offset`
- `roi`

后续可以继续补：

- `structural_alignment_score`
- `stage_confidence`
- `recommended_evidence_type`

## 一阶段落地含义

对于 M400 头戴质检，一阶段不追求让 `YOLO + V-JEPA` 立刻合成单个权重文件，而是先做到：

1. `YOLO` 负责找到控制柜、仪表、游标卡尺等关键目标
2. `V-JEPA` 负责判断当前视频窗口是不是“像一个合格证据片段”
3. `Fusion` 给出最终取景判断
4. `GuidancePolicyService` 再把它翻成“左移 / 右移 / 后退 / 保持”

这样后面即使替换 YOLO 版本或者 V-JEPA 编码器，业务层也不用改。
