# Sensitive Policy Replay Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 补齐 sensitive replay 闭环的工程可用性和判别力，让 CLI 能直接落盘输出，并让 runtime policy 候选不再因样本不足而打平。

**Architecture:** 保持现有 `run-research-step -> run_experiment_batch -> judge` 主链不变，只在 CLI 入口增加统一输出目录创建，并在 `GuidanceBoundaryDatasetBuilder` 中新增一个镜像方向阈值样本。这样研究闭环的核心编排和 DDD 分层都不需要调整。

**Tech Stack:** Python 3.10+, pytest, pathlib, json, existing allinone CLI/runtime/research adapters

---

### Task 1: 让 CLI 自动创建输出父目录

**Files:**
- Modify: `src/allinone/interfaces/cli/main.py`
- Test: `tests/interfaces/cli/test_cli_smoke.py`

**Step 1: Write the failing test**

- 给 `build-observation-payload`
- `detect-image`
- `analyze-clip`

至少补一个 smoke test，验证输出父目录不存在时命令依然能成功写文件。

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -k auto_create_output_parent -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 在 CLI 中增加一个小 helper
- 所有直接写 `output` 文件的命令先创建 `parent`

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -k auto_create_output_parent -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/interfaces/cli/main.py tests/interfaces/cli/test_cli_smoke.py
git commit -m "fix: auto-create cli output directories"
```

### Task 2: 增加第 4 个镜像方向边界样本

**Files:**
- Modify: `src/allinone/infrastructure/research/autoresearch/guidance_boundary_dataset.py`
- Test: `tests/infrastructure/research/test_autoresearch_adapters_behavior.py`
- Modify: `experiments/README.md`

**Step 1: Write the failing test**

- 验证生成结果从 3 个 case 变成 4 个 case
- 验证存在 `reverse_direction_trigger_boundary`
- 验证 baseline 对该 case 输出 `hold_still / stabilize_before_capture`
- 验证 `earlier_direction_trigger` 对该 case 输出 `right / target_shifted_left`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py -k boundary_dataset -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 在 builder 中新增镜像方向边界 case
- 复用现有 bbox 生成逻辑，只改 `center_x`
- 更新 manifest 和 README 示例说明

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py -k boundary_dataset -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/research/autoresearch/guidance_boundary_dataset.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py experiments/README.md
git commit -m "feat: add mirrored direction replay case"
```

### Task 3: 完整验证

**Files:**
- Reuse existing files only

**Step 1: Run targeted tests**

Run:

```bash
python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -k auto_create_output_parent -v
python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py -k boundary_dataset -v
```

Expected: PASS

**Step 2: Run full local suite**

Run: `python3 -m pytest tests -v`
Expected: PASS

**Step 3: Optional remote replay confirmation**

Run on server after sync:

```bash
python3 -m allinone.interfaces.cli.main build-guidance-replay-dataset --input-raw experiments/generated/base_person_raw.json --output-dir experiments/generated/person_boundary_replay --target-label person
python3 -m allinone.interfaces.cli.main run-research-step --experiment-id exp-loop-sensitive-003 --hypothesis "compare sensitive replay boundary cases with mirrored direction" --target-metric guidance_success_rate --manifest experiments/generated/person_boundary_replay/manifest.jsonl --base-policy configs/runtime_policies/m400_default.json --candidate-count 4 --run-root experiments/research/exp-loop-sensitive-003 --output experiments/research/exp-loop-sensitive-003/summary.json --yolo-model /home/dell/yolo11n.pt --vjepa-repo /home/dell/vjepa2 --vjepa-checkpoint /home/dell/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt --sample-frames 4
```

Expected:

- replay dataset 包含 4 个 case
- `candidate-2` 与 `candidate-3` 不再打平
