# Experiment Batch Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `allinone` 增加第一条正式的批量离线回放实验链，让一批 clip 可以被统一执行、归档、统计，并形成后续 `autoresearch` 可消费的实验资产。

**Architecture:** 保持现有单条 clip 主链不变，在其上新增两层：一层是可复用的 runtime 结果用例，把 guidance + Qwen 解释从 CLI 中抽出来；另一层是研究批处理用例，从 manifest 驱动 clip 批量执行，并把 `raw / payload / result / summary` 归档到 run 目录。CLI 只负责接参数和触发实验运行。

**Tech Stack:** Python 3.14, pytest, json/jsonl, pathlib, existing `YOLO + V-JEPA + Qwen` adapters, local filesystem run artifacts

---

### Task 1: 抽出可复用的 runtime 结果用例

**Files:**
- Create: `src/allinone/application/runtime/run_runtime_observation.py`
- Modify: `src/allinone/interfaces/cli/main.py`
- Modify: `tests/application/test_runtime_usecases.py`
- Modify: `tests/application/test_usecase_layout.py`

**Step 1: Write the failing test**

在 `tests/application/test_runtime_usecases.py` 新增测试，验证：

- 输入标准 payload
- 返回结构化 runtime 结果
- 结果里至少包含：
  - `guidance_action`
  - `reason`
  - `language_action`
  - `confidence`
  - `operator_message`
  - `language_source`

同时更新 `tests/application/test_usecase_layout.py`，要求新模块存在。

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/application/test_runtime_usecases.py tests/application/test_usecase_layout.py -v`
Expected: FAIL because `run_runtime_observation.py` does not exist.

**Step 3: Write minimal implementation**

- 创建 `run_runtime_observation.py`
- 把现有 CLI 中的：
  - `ingest_observation_window`
  - `request_guidance_decision`
  - `QwenPromptBuilder`
  - `QwenStructuredOutputParser`
  - `QwenClient`
  组合为一个可复用用例
- CLI 的 `runtime-observation` 命令改为调用新用例

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/application/test_runtime_usecases.py tests/application/test_usecase_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/application/runtime/run_runtime_observation.py src/allinone/interfaces/cli/main.py tests/application/test_runtime_usecases.py tests/application/test_usecase_layout.py
git commit -m "feat: extract runtime observation use case

Co-authored-by: Codex <noreply@openai.com>"
```

### Task 2: 增加批量实验运行用例

**Files:**
- Create: `src/allinone/application/research/run_experiment_batch.py`
- Modify: `tests/application/test_research_usecases.py`
- Modify: `tests/application/test_usecase_layout.py`

**Step 1: Write the failing test**

在 `tests/application/test_research_usecases.py` 新增测试，验证：

- 输入 manifest rows
- 对每条 row 调用 clip 感知和 runtime 用例
- 返回按 clip 聚合的 experiment result rows

测试中使用假 clip analyzer、假 runtime usecase、假 writer，先锁定编排逻辑。

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/application/test_research_usecases.py tests/application/test_usecase_layout.py -v`
Expected: FAIL because `run_experiment_batch.py` does not exist.

**Step 3: Write minimal implementation**

- 创建 `run_experiment_batch.py`
- 提供 `run_experiment_batch(...)`
- 接收：
  - `manifest_rows`
  - `candidate_name`
  - `clip_analyzer`
  - `runtime_runner`
  - `run_writer`
- 输出结构化 `results`

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/application/test_research_usecases.py tests/application/test_usecase_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/application/research/run_experiment_batch.py tests/application/test_research_usecases.py tests/application/test_usecase_layout.py
git commit -m "feat: add experiment batch use case

Co-authored-by: Codex <noreply@openai.com>"
```

### Task 3: 增加 run 目录写入与 summary 统计

**Files:**
- Create: `src/allinone/infrastructure/research/autoresearch/run_writer.py`
- Modify: `src/allinone/infrastructure/research/autoresearch/replay_adapter.py`
- Modify: `tests/infrastructure/research/test_autoresearch_adapters_behavior.py`
- Modify: `tests/infrastructure/research/test_autoresearch_adapter_layout.py`

**Step 1: Write the failing test**

在 `tests/infrastructure/research/test_autoresearch_adapters_behavior.py` 新增测试，验证：

- 能把一次 batch run 写到 run 目录
- 能生成：
  - `manifest.jsonl`
  - `results.jsonl`
  - `summary.json`
- `summary.json` 至少包含：
  - `action_match_rate`
  - `target_detected_rate`
  - `usable_clip_rate`

同时更新 layout test，要求新模块存在。

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py tests/infrastructure/research/test_autoresearch_adapter_layout.py -v`
Expected: FAIL because `run_writer.py` does not exist.

**Step 3: Write minimal implementation**

- 创建 `run_writer.py`
- 实现：
  - 写 manifest
  - 写 results
  - 计算 summary
- 在 `replay_adapter.py` 中补充可消费 run 目录结果的 payload 构造

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/research/test_autoresearch_adapters_behavior.py tests/infrastructure/research/test_autoresearch_adapter_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/research/autoresearch/run_writer.py src/allinone/infrastructure/research/autoresearch/replay_adapter.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py tests/infrastructure/research/test_autoresearch_adapter_layout.py
git commit -m "feat: add experiment run artifact writer

Co-authored-by: Codex <noreply@openai.com>"
```

### Task 4: 增加 `run-experiment` CLI 与示例 manifest

**Files:**
- Modify: `src/allinone/interfaces/cli/main.py`
- Create: `experiments/manifests/m400_phase1_demo.jsonl`
- Modify: `experiments/README.md`
- Modify: `tests/interfaces/cli/test_cli_smoke.py`

**Step 1: Write the failing test**

在 `tests/interfaces/cli/test_cli_smoke.py` 新增测试，验证：

- `run-experiment` 命令存在
- 能接收：
  - `--manifest`
  - `--run-dir`
  - `--candidate`
  - `--yolo-model`
  - `--vjepa-repo`
  - `--vjepa-checkpoint`
- 能写出 run 目录和 summary 文件

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -v`
Expected: FAIL because `run-experiment` command does not exist.

**Step 3: Write minimal implementation**

- 在 CLI parser 中增加 `run-experiment`
- manifest 读取后调用：
  - clip analyzer
  - runtime usecase
  - run writer
- 增加一个最小示例 manifest
- 补充 `experiments/README.md`

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/interfaces/cli/main.py experiments/manifests/m400_phase1_demo.jsonl experiments/README.md tests/interfaces/cli/test_cli_smoke.py
git commit -m "feat: add run-experiment cli entrypoint

Co-authored-by: Codex <noreply@openai.com>"
```

### Task 5: 跑本地与服务器完整验证

**Files:**
- Reuse existing files only

**Step 1: Run local targeted tests**

Run:

```bash
python3 -m pytest tests/application/test_runtime_usecases.py tests/application/test_research_usecases.py tests/infrastructure/research/test_autoresearch_adapters_behavior.py tests/interfaces/cli/test_cli_smoke.py -v
```

Expected: PASS

**Step 2: Run full local test suite**

Run:

```bash
python3 -m pytest tests -v
```

Expected: PASS

**Step 3: Sync to server and verify tests**

Run:

```bash
bash ops/remote/sync_to_server.sh
ssh dell@192.168.1.104 'cd /home/dell/workspaces/allinone && . .venv/bin/activate && export PYTHONPATH=/home/dell/workspaces/allinone:/home/dell/workspaces/allinone/src && python3 -m pytest tests -v'
```

Expected: PASS

**Step 4: Run a real server experiment batch**

准备一个最小 manifest，至少包含 2 到 3 条 clip。

执行：

```bash
python3 -m allinone.interfaces.cli.main run-experiment \
  --manifest experiments/manifests/m400_phase1_demo.jsonl \
  --run-dir experiments/runs/run-2026-03-23-m400-phase1 \
  --candidate baseline \
  --yolo-model /home/dell/yolo11n.pt \
  --vjepa-repo /home/dell/vjepa2 \
  --vjepa-checkpoint /home/dell/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
```

验证产物：

- `manifest.jsonl`
- `results.jsonl`
- `summary.json`

并检查 `summary.json` 中至少包含：

- `action_match_rate`
- `target_detected_rate`
- `usable_clip_rate`

**Step 5: Commit**

```bash
git add .
git commit -m "test: verify experiment batch replay loop

Co-authored-by: Codex <noreply@openai.com>"
```
