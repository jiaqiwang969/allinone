# Autoresearch Mapping

## 目标

把旧仓 `29-autoresearch/autoresearch` 中已经存在的自动实验能力，收编进 `allinone` 的 `research` 边界上下文里。这里的核心不是直接搬文件，而是把旧项目的“实验循环”翻译成围绕 `ExperimentRun` 的统一研究控制面。

## 核心映射

| 旧资产 | 新位置 | 新职责 |
| --- | --- | --- |
| `autoresearch/runtime_config.py` | `infrastructure/research/autoresearch/replay_adapter.py` | 读取实验运行配置，驱动数据回放与批量评估 |
| `autoresearch/train.py` | `application/research` + `infrastructure/research/autoresearch` | 作为实验执行入口，被新的用例层调度 |
| `autoresearch/prepare.py` | `bootstrap` / `configs/data_recipes` | 预处理流程改为对统一数据本体做准备 |
| `autoresearch/program.md` | `docs/architecture` / `docs/plans` | 保留为实验假设、 judge 规则和闭环说明 |
| `autoresearch/results.tsv` | `research` 仓储输出 | 归档为 `ExperimentRun` 的评估结果快照 |
| `autoresearch/analysis.ipynb` | `experiments/` | 只保留分析角色，不直接进入核心域 |

## 在新架构里的位置

- `domain.research`
  - 定义 `ExperimentRun`、评估状态、指标快照、候选配置
- `application.research`
  - 定义“注册实验”“启动回放”“触发 judge”“记录结果”的用例
- `infrastructure.research.autoresearch`
  - 负责兼容旧的 autoresearch 代码、脚本和输出格式

## 为什么这样做

旧 `autoresearch` 擅长的是自动试验、打分、选优，但它不知道工业作业 session、取景提示、证据是否合格。`allinone` 的做法是：

1. 运行时先产生 `session -> guidance -> evidence` 的业务结果
2. 再把这些结果喂给 `ExperimentRun`
3. 由 autoresearch adapter 回放不同策略、不同模型组合
4. 输出下一轮该改什么阈值、提示词、数据选择和训练 recipe

这就把旧 autoresearch 从“单独试验脚本”变成了整个闭环里的研究引擎。
