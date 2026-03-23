# Candidate Judge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `allinone` 增加第一版多 candidate 规则 judge 链，使多个 `run_dir` 可以被统一打分、比较并选出最佳 candidate。

**Architecture:** 在现有 `run-experiment -> run_dir` 基础上，新增 `application.research` judge 用例，调用 `replay_adapter + rule_based_judge + judge_adapter + ExperimentSelectionService`。CLI 只负责参数解析与结果输出。

**Tech Stack:** Python 3.14, pytest, json/jsonl, pathlib, existing research domain entities/services

---

### Task 1: 扩展 replay adapter 支持 judge 输入

**Files:**
- Modify: `src/allinone/infrastructure/research/autoresearch/replay_adapter.py`
- Modify: `tests/infrastructure/research/test_autoresearch_adapters_behavior.py`

**Step 1: Write the failing test**

在 `tests/infrastructure/research/test_autoresearch_adapters_behavior.py` 增加断言：

- `build_run_payload(run_dir)` 返回 `results_path`
- 保留 `run_dir / candidate_name / summary / result_count`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py -k run_directory -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 在 `build_run_payload` 中增加 `results_path`

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py -k run_directory -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/research/autoresearch/replay_adapter.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py
git commit -m "feat: extend replay payload for candidate judge"
```

### Task 2: 增加规则 judge 基础设施

**Files:**
- Create: `src/allinone/infrastructure/research/autoresearch/rule_based_judge.py`
- Modify: `tests/infrastructure/research/test_autoresearch_adapter_layout.py`
- Modify: `tests/infrastructure/research/test_autoresearch_adapters_behavior.py`

**Step 1: Write the failing test**

新增测试，验证：

- 读取 `summary.json + results.jsonl`
- 输出 `score`
- 输出简短 `summary`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py tests/infrastructure/research/test_autoresearch_adapter_layout.py -k judge -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 创建 `RuleBasedAutoresearchJudge`
- 实现主分 + 惩罚项

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py tests/infrastructure/research/test_autoresearch_adapter_layout.py -k judge -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/research/autoresearch/rule_based_judge.py tests/infrastructure/research/test_autoresearch_adapter_layout.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py
git commit -m "feat: add rule-based candidate judge"
```

### Task 3: 增加 judge experiment application use case

**Files:**
- Create: `src/allinone/application/research/judge_experiment_candidates.py`
- Modify: `tests/application/test_research_usecases.py`
- Modify: `tests/application/test_usecase_layout.py`

**Step 1: Write the failing test**

新增测试，验证：

- 输入 experiment metadata + candidate run list
- 注册 `ExperimentRun`
- 记录每个 candidate 的 evaluation
- 选出 `best_candidate_name`
- 返回结构化结果

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/application/test_research_usecases.py tests/application/test_usecase_layout.py -k judge -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 创建 `judge_experiment_candidates.py`
- 编排 replay/judge/adapter/domain selection

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/application/test_research_usecases.py tests/application/test_usecase_layout.py -k judge -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/application/research/judge_experiment_candidates.py tests/application/test_research_usecases.py tests/application/test_usecase_layout.py
git commit -m "feat: add experiment candidate judge use case"
```

### Task 4: 增加 judge-experiment CLI

**Files:**
- Modify: `src/allinone/interfaces/cli/main.py`
- Modify: `tests/interfaces/cli/test_cli_smoke.py`
- Modify: `experiments/README.md`

**Step 1: Write the failing test**

新增测试，验证：

- `judge-experiment` 命令存在
- 能接受多个 `--candidate-run`
- 能写出 `judgement.json`
- 输出 `best_candidate_name`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -k judge_experiment -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 解析 `candidate=run_dir`
- 调用 judge use case
- 将结果写到 `--output`

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -k judge_experiment -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/interfaces/cli/main.py tests/interfaces/cli/test_cli_smoke.py experiments/README.md
git commit -m "feat: add judge-experiment cli"
```

### Task 5: 做本地与服务器验证

**Files:**
- Reuse existing files only

**Step 1: Run local full tests**

Run: `python3 -m pytest tests -v`
Expected: PASS

**Step 2: Sync to server and run tests**

Run:

```bash
RSYNC_RSH="sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no" bash ops/remote/sync_to_server.sh
sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no dell@192.168.1.104 'cd /home/dell/workspaces/allinone && . .venv/bin/activate && export PYTHONPATH=/home/dell/workspaces/allinone:/home/dell/workspaces/allinone/src && python3 -m pytest tests -v'
```

Expected: PASS

**Step 3: Run a real server judge**

基于服务器已有的多个 `run_dir` 或伪造 demo run：

```bash
python3 -m allinone.interfaces.cli.main judge-experiment \
  --experiment-id exp-judge-001 \
  --hypothesis "compare candidate runs" \
  --target-metric guidance_success_rate \
  --candidate-run baseline=experiments/runs/run-a \
  --candidate-run candidate-a=experiments/runs/run-b \
  --output experiments/judgements/exp-judge-001.json
```

Expected:

- 成功输出 `best_candidate_name`
- 成功写出 judgement JSON
