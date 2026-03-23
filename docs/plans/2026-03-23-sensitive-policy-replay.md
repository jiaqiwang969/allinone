# Sensitive Policy Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 runtime policy research loop 增加敏感 replay 数据体系，让实验既支持真实 clip，也支持冻结 raw payload，并让 judge 能观测 reason 差异。

**Architecture:** 在现有 `run_experiment_batch -> run_research_step -> judge_experiment_candidates` 主链上，增加 `raw_payload_path` replay 输入、边界 raw payload 生成器，以及可选 `reason_match_rate` 评分逻辑。这样 clip 分析只需要做一次，后续 policy 搜索复用冻结的 perception trace。

**Tech Stack:** Python 3.10+, pytest, json, pathlib, existing YOLO/V-JEPA/Qwen adapters, existing research CLI

---

### Task 1: 让 research replay 支持 `raw_payload_path`

**Files:**
- Create: `src/allinone/infrastructure/research/autoresearch/raw_payload_loader.py`
- Modify: `src/allinone/application/research/run_experiment_batch.py`
- Modify: `src/allinone/interfaces/cli/main.py`
- Modify: `tests/application/test_research_usecases.py`
- Modify: `tests/interfaces/cli/test_cli_smoke.py`

**Step 1: Write the failing test**

- 增加测试验证：
  - manifest row 提供 `raw_payload_path` 时，不再调用 `clip_analyzer`
  - runtime replay 仍能正确写出 run artifacts

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/application/test_research_usecases.py tests/interfaces/cli/test_cli_smoke.py -k raw_payload_path -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 增加 `RawPayloadLoader`
- `run_experiment_batch` 支持两种输入：
  - `clip_path`
  - `raw_payload_path`
- CLI 装配 loader

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/application/test_research_usecases.py tests/interfaces/cli/test_cli_smoke.py -k raw_payload_path -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/research/autoresearch/raw_payload_loader.py src/allinone/application/research/run_experiment_batch.py src/allinone/interfaces/cli/main.py tests/application/test_research_usecases.py tests/interfaces/cli/test_cli_smoke.py
git commit -m "feat: support raw payload replay inputs"
```

### Task 2: 给 runtime result / summary / judge 增加 `reason_match`

**Files:**
- Modify: `src/allinone/application/research/run_experiment_batch.py`
- Modify: `src/allinone/infrastructure/research/autoresearch/run_writer.py`
- Modify: `src/allinone/infrastructure/research/autoresearch/rule_based_judge.py`
- Modify: `tests/application/test_research_usecases.py`
- Modify: `tests/infrastructure/research/test_autoresearch_adapters_behavior.py`

**Step 1: Write the failing test**

- 增加测试验证：
  - manifest row 有 `expected_reason` 时，result row 记录：
    - `guidance_reason`
    - `expected_reason`
    - `reason_match`
  - summary 输出 `reason_match_rate`
  - judge 在存在 reason 监督时把 `reason_match_rate` 纳入评分

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/application/test_research_usecases.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py -k reason_match -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 将 runtime `reason` 写入 result row
- summary 增加 `reason_match_rate`
- judge 保持向后兼容：
  - 有 `reason_match_rate` 就纳入评分
  - 没有就沿用旧逻辑

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/application/test_research_usecases.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py -k reason_match -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/application/research/run_experiment_batch.py src/allinone/infrastructure/research/autoresearch/run_writer.py src/allinone/infrastructure/research/autoresearch/rule_based_judge.py tests/application/test_research_usecases.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py
git commit -m "feat: score reason-aware replay results"
```

### Task 3: 增加边界 replay 数据生成器

**Files:**
- Create: `src/allinone/infrastructure/research/autoresearch/guidance_boundary_dataset.py`
- Modify: `tests/infrastructure/research/test_autoresearch_adapter_layout.py`
- Modify: `tests/infrastructure/research/test_autoresearch_adapters_behavior.py`

**Step 1: Write the failing test**

- 验证输入 base raw payload 后可以写出：
  - `tight_center_boundary`
  - `direction_trigger_boundary`
  - `oversize_boundary`
- 同时生成 `manifest.jsonl`
- manifest row 包含：
  - `raw_payload_path`
  - `expected_action`
  - `expected_reason`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapter_layout.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py -k boundary_dataset -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 以 base detection bbox 为模板
- 用固定阈值带生成三类边界 case
- 写出 raw payload files 和 manifest

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapter_layout.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py -k boundary_dataset -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/research/autoresearch/guidance_boundary_dataset.py tests/infrastructure/research/test_autoresearch_adapter_layout.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py
git commit -m "feat: add guidance boundary replay dataset builder"
```

### Task 4: 增加 `build-guidance-replay-dataset` CLI

**Files:**
- Modify: `src/allinone/interfaces/cli/main.py`
- Modify: `experiments/README.md`
- Modify: `tests/interfaces/cli/test_cli_smoke.py`

**Step 1: Write the failing test**

- 验证命令存在
- 能读取 base raw payload
- 能写出 replay dataset 目录和 manifest

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -k guidance_replay_dataset -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- CLI 参数解析
- 装配 dataset builder
- 输出生成结果摘要

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -k guidance_replay_dataset -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/interfaces/cli/main.py experiments/README.md tests/interfaces/cli/test_cli_smoke.py
git commit -m "feat: add guidance replay dataset cli"
```

### Task 5: 本地和服务器生成“不打平”的真实 demo

**Files:**
- Reuse existing files only

**Step 1: Run local full tests**

Run: `python3 -m pytest tests -v`
Expected: PASS

**Step 2: Sync to server and run full tests**

Run:

```bash
RSYNC_RSH="sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no" bash ops/remote/sync_to_server.sh
sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no dell@192.168.1.104 'cd /home/dell/workspaces/allinone && . .venv/bin/activate && export PYTHONPATH=/home/dell/workspaces/allinone:/home/dell/workspaces/allinone/src && python3 -m pytest tests -v'
```

Expected: PASS

**Step 3: Generate boundary replay dataset from one real clip**

Run:

```bash
sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no dell@192.168.1.104 'cd /home/dell/workspaces/allinone && . .venv/bin/activate && export PYTHONPATH=/home/dell/workspaces/allinone:/home/dell/workspaces/allinone/src && python3 -m allinone.interfaces.cli.main analyze-clip --clip /home/dell/.cache/.fr-oD2sQN/factory001_worker001_00000.mp4 --yolo-model /home/dell/yolo11n.pt --vjepa-repo /home/dell/vjepa2 --vjepa-checkpoint /home/dell/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt --targets person --output experiments/generated/base_person_raw.json --sample-frames 4 && python3 -m allinone.interfaces.cli.main build-guidance-replay-dataset --input-raw experiments/generated/base_person_raw.json --output-dir experiments/generated/person_boundary_replay --target-label person'
```

Expected:

- 生成三类边界 raw payload
- 自动写出 manifest

**Step 4: Run a real research step on the generated manifest**

Run:

```bash
sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no dell@192.168.1.104 'cd /home/dell/workspaces/allinone && . .venv/bin/activate && export PYTHONPATH=/home/dell/workspaces/allinone:/home/dell/workspaces/allinone/src && python3 -m allinone.interfaces.cli.main run-research-step --experiment-id exp-loop-sensitive-001 --hypothesis "compare sensitive replay boundary cases" --target-metric guidance_success_rate --manifest experiments/generated/person_boundary_replay/manifest.jsonl --base-policy configs/runtime_policies/m400_default.json --candidate-count 3 --run-root experiments/research/exp-loop-sensitive-001 --output experiments/research/exp-loop-sensitive-001/summary.json --yolo-model /home/dell/yolo11n.pt --vjepa-repo /home/dell/vjepa2 --vjepa-checkpoint /home/dell/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt --sample-frames 4'
```

Expected:

- 不再所有 candidate 打平
- `summary.json` / `judgement.json` 成功写出
- 至少一个 mutation 在 action 或 reason 指标上体现差异
