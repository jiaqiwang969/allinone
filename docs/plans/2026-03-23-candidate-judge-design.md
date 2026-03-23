# Candidate Judge 设计

## 背景

当前 `allinone` 已经具备：

1. 单 clip 推理链
2. 批量 manifest 回放
3. `run_dir -> manifest/results/summary/raw/payload` 落盘
4. 服务器真实样本回放验证

现在缺的不是“再跑一批”，而是“多批结果如何比较，并自动选出更好的 candidate”。

这一步补上以后，`autoresearch` 的闭环才算真正开始成立：

`candidate 配置 -> run-experiment -> run_dir -> judge -> best candidate`

## 目标

新增一条最小可用的研究 judge 链：

- 输入多个 `run_dir`
- 每个 `run_dir` 对应一个 `candidate_name`
- 读取 `summary.json + results.jsonl`
- 生成规则化评分
- 写回结构化比较结果
- 选出 `best candidate`

第一版只做离线、稳定、可复现的规则 judge。

## 为什么第一版不用 LLM judge

原因很直接：

1. 现在更需要可复现，不是“更聪明”
2. `summary.json` 已经有稳定指标
3. `results.jsonl` 已经能提供错误和漏检信息
4. 规则 judge 便于回放、对比、调权重
5. LLM judge 后面可以作为增强层再接入

所以第一版 judge 的职责不是“理解复杂业务语义”，而是先把“实验比较控制面”打通。

## Judge 规则

### 主分

每个 candidate 的主分先来自 `summary.json`：

- `action_match_rate`
- `target_detected_rate`
- `usable_clip_rate`

建议默认权重：

- `0.5 * action_match_rate`
- `0.3 * target_detected_rate`
- `0.2 * usable_clip_rate`

### 惩罚项

再从 `results.jsonl` 增加两个轻惩罚：

- `error_rate`
- `target_not_detected_ratio`

惩罚只做“小幅拉开”，不覆盖主分。

建议：

- `error_penalty = 0.1 * error_rate`
- `missing_target_penalty = 0.05 * target_not_detected_ratio`

最终：

`score = max(0.0, main_score - error_penalty - missing_target_penalty)`

## 架构落点

### `application.research`

新增一个 judge 用例，负责：

- 注册 `ExperimentRun`
- 读取每个 candidate 的 `run_dir`
- 调用 replay adapter 和规则 judge
- 记录 `CandidateEvaluation`
- 调用选择服务选出 best candidate
- 完成实验

### `infrastructure.research.autoresearch`

补一个规则 judge 组件，负责：

- 读取 `summary.json`
- 读取 `results.jsonl`
- 计算评分
- 生成简短 summary

同时扩展 replay adapter，让它返回：

- `run_dir`
- `summary`
- `results_path`
- `result_count`

### `interfaces.cli`

新增 `judge-experiment` 命令。

建议参数：

- `--experiment-id`
- `--hypothesis`
- `--target-metric`
- `--candidate-run baseline=/path/to/run1`
- `--candidate-run candidate-a=/path/to/run2`
- `--output /path/to/judgement.json`

CLI 只做参数解析、触发 use case、写输出。

## 输出格式

Judge 输出一个结构化 JSON：

- `experiment_id`
- `target_metric`
- `candidate_scores`
- `best_candidate_name`
- `status`

其中 `candidate_scores` 至少包含：

- `candidate_name`
- `run_dir`
- `score`
- `summary`

## 非目标

本轮不做：

- LLM judge
- 自动再次触发新一轮 `run-experiment`
- candidate 并行执行调度
- 复杂多目标打分体系
- 在线学习或训练

## 成功标准

本轮完成后，应满足：

1. 能基于多个 `run_dir` 形成结构化评估
2. 能生成稳定、可复现的 candidate score
3. 能自动选出 `best candidate`
4. 本地和服务器都能跑通
5. 输出结果能被后续 autoresearch 闭环继续消费
