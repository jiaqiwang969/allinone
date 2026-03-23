# Runtime Policy Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `allinone` 增加第一版真正可执行的 autoresearch step，让 candidate policy 能真实改变 guidance 行为，并完成 `生成 candidate -> 回放 -> judge -> 选优` 单步闭环。

**Architecture:** 在 `domain.guidance` 现有阈值规则之上，新增 `infrastructure.guidance` recipe 读写与 mutation proposer；`application.research` 新增单步 research loop 用例；CLI 提供 `run-research-step` 入口，顺序调用现有 `run_experiment_batch` 与 `judge_experiment_candidates`。

**Tech Stack:** Python 3.14, pytest, json, pathlib, existing YOLO/V-JEPA/Qwen adapters, existing research judge chain

---

### Task 1: 让 runtime 能读取 policy recipe 并影响 guidance 决策

**Files:**
- Create: `src/allinone/infrastructure/guidance/policy_recipe.py`
- Modify: `src/allinone/application/runtime/request_guidance_decision.py`
- Modify: `src/allinone/application/runtime/run_runtime_observation.py`
- Modify: `tests/application/test_runtime_usecases.py`
- Modify: `tests/infrastructure/perception/test_perception_adapter_layout.py`

**Step 1: Write the failing test**

- 增加测试验证：
  - 可从 JSON recipe 读取 `GuidanceThresholds`
  - `run_runtime_observation` 传入 policy thresholds 后会改变 `guidance_action`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/application/test_runtime_usecases.py -k policy -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 增加 recipe loader
- 让 runtime observation 接收可选 `guidance_thresholds`

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/application/test_runtime_usecases.py -k policy -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/guidance/policy_recipe.py src/allinone/application/runtime/request_guidance_decision.py src/allinone/application/runtime/run_runtime_observation.py tests/application/test_runtime_usecases.py tests/infrastructure/perception/test_perception_adapter_layout.py
git commit -m "feat: add runtime policy recipe support"
```

### Task 2: 增加 rule-based candidate policy proposer

**Files:**
- Create: `src/allinone/infrastructure/research/autoresearch/policy_candidate_proposer.py`
- Modify: `tests/infrastructure/research/test_autoresearch_adapter_layout.py`
- Modify: `tests/infrastructure/research/test_autoresearch_adapters_behavior.py`

**Step 1: Write the failing test**

- 验证基于 base policy 可以生成：
  - `baseline`
  - 若干 mutation candidate
- 每个 candidate 都有独立 policy 内容

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py tests/infrastructure/research/test_autoresearch_adapter_layout.py -k proposer -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 实现固定 mutation 列表
- 返回结构化 candidate policy rows

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py tests/infrastructure/research/test_autoresearch_adapter_layout.py -k proposer -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/research/autoresearch/policy_candidate_proposer.py tests/infrastructure/research/test_autoresearch_adapter_layout.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py
git commit -m "feat: add policy candidate proposer"
```

### Task 3: 增加 application research step 用例

**Files:**
- Create: `src/allinone/application/research/run_research_step.py`
- Modify: `tests/application/test_research_usecases.py`
- Modify: `tests/application/test_usecase_layout.py`

**Step 1: Write the failing test**

- 输入：
  - manifest rows
  - base policy
  - candidate count
- 验证：
  - materialize candidate policies
  - 每个 candidate 执行 experiment
  - 自动 judge
  - 返回 best policy summary

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/application/test_research_usecases.py tests/application/test_usecase_layout.py -k research_step -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- 顺序编排 proposer / batch runner / judge
- 写出结构化 summary

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/application/test_research_usecases.py tests/application/test_usecase_layout.py -k research_step -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/application/research/run_research_step.py tests/application/test_research_usecases.py tests/application/test_usecase_layout.py
git commit -m "feat: add research step use case"
```

### Task 4: 增加 `run-research-step` CLI

**Files:**
- Modify: `src/allinone/interfaces/cli/main.py`
- Modify: `tests/interfaces/cli/test_cli_smoke.py`
- Modify: `experiments/README.md`

**Step 1: Write the failing test**

- 验证命令存在
- 能读取 base policy
- 能写出 summary JSON
- stdout 含 `best_candidate_name`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -k research_step -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- CLI 参数解析
- 装配 recipe loader / proposer / runner / judge
- 写出结果 JSON

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -k research_step -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/interfaces/cli/main.py tests/interfaces/cli/test_cli_smoke.py experiments/README.md
git commit -m "feat: add run-research-step cli"
```

### Task 5: 本地与服务器验证

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

**Step 3: Run one real research step on server**

Run:

```bash
python3 -m allinone.interfaces.cli.main run-research-step \
  --experiment-id exp-loop-001 \
  --hypothesis "tighten guidance thresholds" \
  --target-metric guidance_success_rate \
  --manifest experiments/manifests/judge_baseline.jsonl \
  --base-policy configs/runtime_policies/m400_default.json \
  --candidate-count 3 \
  --run-root experiments/research/exp-loop-001 \
  --output experiments/research/exp-loop-001/summary.json \
  --yolo-model /home/dell/yolo11n.pt \
  --vjepa-repo /home/dell/vjepa2 \
  --vjepa-checkpoint /home/dell/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
```

Expected:

- 成功写出 candidate policies
- 成功写出 candidate run dirs
- 成功写出 judgement
- 成功给出 `best_candidate_name`
