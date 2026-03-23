# Runtime Policy Loop 设计

## 背景

当前 `allinone` 已经有两条链：

1. `run-experiment`：把一批 clip 回放成 `run_dir`
2. `judge-experiment`：把多个 `run_dir` 评分并选出最优 candidate

但这还不是 `autoresearch` 的闭环。原因很简单：

- 现在的 `candidate` 主要只是一个名字
- 它还没有真正改变运行时行为
- 所以即使能 judge，也还没有稳定的“动作面”

站在第一性原理上，`autoresearch` 能自动化，必须同时具备：

1. 明确目标函数
2. 固定裁判
3. 可受控的动作空间
4. 外部记忆
5. keep / discard 规则

前两项已经有了，第三项还缺。

## 目标

补齐第一版真正可执行的研究动作面：

- 把 `guidance` 阈值外置成 `runtime policy recipe`
- 让 `run-experiment` 可以按 recipe 运行
- 基于当前最优 recipe 自动生成若干 candidate recipe
- 一次命令内完成：
  `生成 candidate -> 执行 replay -> judge -> 选优 -> 写回 frontier`

第一版只做 `guidance threshold recipe`，不做模型权重训练。

## 方案比较

### 方案 A：继续只用 candidate 名字

- 优点：最快
- 缺点：实验没有真实动作面，judge 只能比较标签，不是真闭环

这个方案应直接放弃。

### 方案 B：引入 runtime policy recipe

- 把 `GuidanceThresholds` 外置为 JSON recipe
- candidate 的差异体现在阈值变化
- `run-experiment` 读取不同 recipe，得到不同 `guidance_action`
- `judge` 再基于结果选优

优点：

- 成本低
- 变化真实可控
- 能立刻进入 autoresearch 闭环
- 完全符合 DDD 边界

缺点：

- 当前只优化规则层，还没触及模型层

这是第一版推荐方案。

### 方案 C：直接做模型微调闭环

- 每轮都改 YOLO / V-JEPA / Qwen 权重

优点：

- 终局价值高

缺点：

- 数据、训练、验证、资源治理都还没准备好
- 当前工程阶段不适合直接上

这个方案留到后面。

## 设计结论

采用方案 B。

第一版 research loop 的真实动作面定义为：

- `centered_offset_max`
- `directional_offset_min`
- `ready_fill_ratio_max`

后续可再扩展：

- prompt recipe
- evidence acceptance rule
- sampling policy
- detector threshold

## 架构落点

### `domain.guidance`

保留 `GuidanceThresholds` 作为领域规则对象。

领域层只表达：

- 有哪些阈值
- 默认值是什么
- 如何基于阈值做决策

领域层不负责：

- 从文件加载 recipe
- 生成 candidate 变体

### `infrastructure.guidance`

新增 recipe loader / writer，负责：

- 从 JSON 读取 policy recipe
- 转成 `GuidanceThresholds`
- 将 candidate recipe 写到实验目录

### `application.research`

新增 research step 用例，负责：

1. 读取基线 policy
2. 生成 baseline + mutated candidates
3. 对每个 candidate 执行 `run_experiment_batch`
4. 对结果调用 `judge_experiment_candidates`
5. 产出 loop artifact
6. 指出 `best_candidate`

### `interfaces.cli`

新增命令：

`run-research-step`

参数建议：

- `--experiment-id`
- `--hypothesis`
- `--target-metric`
- `--manifest`
- `--base-policy`
- `--candidate-count`
- `--run-root`
- `--output`
- `--yolo-model`
- `--vjepa-repo`
- `--vjepa-checkpoint`
- `--device`
- `--sample-frames`

## Candidate 生成策略

第一版不用 LLM 生成 candidate。

直接用规则 mutation：

- baseline：原 recipe
- candidate-1：减小 `centered_offset_max`
- candidate-2：减小 `directional_offset_min`
- candidate-3：增大 `ready_fill_ratio_max`

如果 `candidate-count` 更大，就按固定 mutation 列表循环展开。

这样有三个好处：

1. 可复现
2. 容易解释
3. 方便后面替换成 LLM proposer

## 输出资产

`run-research-step` 至少写出：

- `candidate_policies/<candidate>.json`
- `runs/<candidate>/...`
- `judgement.json`
- `summary.json`

其中 `summary.json` 记录：

- `experiment_id`
- `candidate_count`
- `best_candidate_name`
- `best_policy_path`
- `judgement_path`

## 非目标

本轮不做：

- 模型权重训练
- LLM candidate proposer
- 多轮自动循环守护进程
- keep/discard 的长期 frontier 历史库
- 在线业务策略热更新

## 成功标准

本轮完成后，应满足：

1. `candidate` 不再只是名字，而是真正能改变 guidance 行为
2. 同一批 manifest 可以自动跑多个 policy candidate
3. 系统能自动 judge 并给出最优 policy
4. 输出结果可被下一轮 autoresearch 继续消费
5. 本地和服务器都能跑通
