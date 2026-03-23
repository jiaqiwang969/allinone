# Sensitive Policy Replay Hardening 设计

## 背景

当前敏感 replay 闭环已经成立，但真实服务器验证暴露出两个剩余问题：

1. `analyze-clip` 等 CLI 写文件时不会自动创建父目录，真实跑批需要额外手工 `mkdir -p`
2. 现有边界数据集只有 3 个 case，`candidate-2` 和 `candidate-3` 各赢 1 个 case，最终打平

这两个问题都不是主架构问题，而是“闭环是否真正好用、能不能稳定区分候选策略”的工程硬伤。

## 目标

补齐第一版 sensitive replay 的工程可用性和判别力：

- CLI 输出路径自动建目录，保证真实命令可以直接执行
- replay 数据集增加第 4 个镜像方向边界样本
- 让 `earlier_direction_trigger` 的收益不只在单侧样本上体现
- 不改动 DDD 分层，不引入新的运行时依赖

## 方案比较

### 方案 A：只补 CLI 自动建目录

优点：

- 风险低
- 真实命令体验更顺

缺点：

- `candidate-2 / candidate-3` 持平问题仍然存在
- research loop 的“选优能力”仍然偏弱

### 方案 B：CLI 自动建目录 + 第 4 个镜像方向边界样本

优点：

- 同时解决“能跑”和“能分出高低”
- 样本仍然是确定性的，不引入额外噪声
- 与现有 threshold mutation 完全对齐

缺点：

- 需要更新测试、数据集 builder 和文档

### 方案 C：直接引入更复杂的多目标/多阶段边界 case

优点：

- 长期覆盖更丰富

缺点：

- 当前阶段属于过度设计
- 会模糊这次修改的目的，增加调试成本

## 设计结论

采用方案 B。

本次 hardening 只做两个最小变化：

1. 给所有直接写 `output` 文件的 CLI 命令统一补父目录创建
2. 在 `GuidanceBoundaryDatasetBuilder` 中增加 `reverse_direction_trigger_boundary`

## 为什么第 4 个 case 选“镜像方向触发”

从第一性原理看，`earlier_direction_trigger` 改的是“方向提示触发阈值”，它天然是左右对称规则。

当前只有一个 `direction_trigger_boundary`：

- baseline：`hold_still / stabilize_before_capture`
- candidate-2：`left / target_shifted_right`

这只能证明“右偏场景”下方向阈值变更有效，但不能证明该变更在对称的“左偏场景”也有效。

增加镜像 case 后：

- baseline：仍然 `hold_still / stabilize_before_capture`
- candidate-2：变成 `right / target_shifted_left`
- candidate-3：保持 baseline 行为

这样可以自然打破 `candidate-2 / candidate-3` 的持平，同时更符合方向策略本身的对称性。

## 变更落点

### CLI

修改 `src/allinone/interfaces/cli/main.py`

- 为 `detect-image`
- `analyze-clip`
- `build-observation-payload`

增加统一的输出父目录创建逻辑。

### Replay Dataset

修改 `src/allinone/infrastructure/research/autoresearch/guidance_boundary_dataset.py`

新增：

- `reverse_direction_trigger_boundary`

其目标是让目标框在水平方向上满足：

- `dx` 落在 `-0.18 < dx <= -0.153`

从而：

- baseline 不触发方向提示
- `earlier_direction_trigger` 触发 `right`

### 测试

需要补两类测试：

1. CLI smoke test
   - 输出文件的父目录不存在时也能成功写出
2. boundary dataset behavior test
   - 新数据集包含 4 个 case
   - 新增 case 在 baseline 和 candidate-2 下产生预期分歧

## 验证标准

本地验证：

- 针对性 pytest 先红后绿
- 完整 `python3 -m pytest tests -v`

后续服务器验证：

- 重新生成 replay dataset
- 再跑一次 `candidate_count=4`
- 目标是让 `candidate-2` 与 `candidate-3` 不再打平
