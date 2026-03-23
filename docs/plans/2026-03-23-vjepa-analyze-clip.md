# V-JEPA Analyze Clip Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `allinone` 增加 `analyze-clip` 入口，把短视频 clip 通过 `YOLO + V-JEPA` 变成统一的 raw perception JSON，并接入现有运行主链。

**Architecture:** 保持现有 `raw perception -> standardized payload -> runtime-observation` 不变，在上游增加 `clip sampler`、`YOLO best-frame` 和 `V-JEPA clip scoring` 三段。`YOLO` 负责几何定位，`V-JEPA` 负责时序质量评分，融合结果继续通过 raw payload 进入业务链。

**Tech Stack:** Python 3.14, pytest, Pillow, OpenCV or decord, torch, local V-JEPA repo, Ultralytics YOLO

---

### Task 1: 加入 clip 采样与 raw payload 用例测试

**Files:**
- Create: `src/allinone/application/runtime/build_clip_perception_payload.py`
- Modify: `tests/application/test_runtime_usecases.py`
- Modify: `tests/application/test_usecase_layout.py`

**Step 1: Write the failing test**

新增测试，验证：

- 输入 clip 路径和假 sampler
- sampler 返回若干采样帧与帧索引
- 用例可以消费采样结果并组织 clip 级 raw payload

测试里使用假 sampler、假 detector、假 V-JEPA scorer，确保先锁定编排逻辑。

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/application/test_runtime_usecases.py tests/application/test_usecase_layout.py -v`
Expected: FAIL because `build_clip_perception_payload.py` does not exist.

**Step 3: Write minimal implementation**

- 创建 `build_clip_perception_payload.py`
- 提供 `build_raw_perception_payload_from_clip(...)`
- 接收 sampler / detector / scorer 注入
- 输出 clip 级 raw JSON

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/application/test_runtime_usecases.py tests/application/test_usecase_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/application/runtime/build_clip_perception_payload.py tests/application/test_runtime_usecases.py tests/application/test_usecase_layout.py
git commit -m "feat: add clip perception use case"
```

### Task 2: 增加 clip sampler 和 V-JEPA scoring adapter

**Files:**
- Create: `src/allinone/infrastructure/perception/video/sampler.py`
- Modify: `src/allinone/infrastructure/perception/vjepa/encoder.py`
- Modify: `tests/infrastructure/perception/test_vjepa_adapter_behavior.py`
- Create: `tests/infrastructure/perception/test_video_sampler_behavior.py`

**Step 1: Write the failing test**

新增测试，验证：

- clip sampler 能从视频中稳定采样固定帧数
- `VJEPAEncoderAdapter` 能提供 `score_clip(...)`
- `score_clip(...)` 至少返回 `visibility_score/readable_ratio/stability_score/alignment_score`

先用假实现和固定输入定义接口，不要求一开始就跑真实模型。

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/perception/test_vjepa_adapter_behavior.py tests/infrastructure/perception/test_video_sampler_behavior.py -v`
Expected: FAIL because new methods and files do not exist.

**Step 3: Write minimal implementation**

- 新增 `ClipFrameSampler`
- 在 `VJEPAEncoderAdapter` 中加入 clip scoring 边界
- 先实现可测试的最小版本
- 为真实 V-JEPA 增加 repo/checkpoint 参数入口

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/perception/test_vjepa_adapter_behavior.py tests/infrastructure/perception/test_video_sampler_behavior.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/perception/video/sampler.py src/allinone/infrastructure/perception/vjepa/encoder.py tests/infrastructure/perception/test_vjepa_adapter_behavior.py tests/infrastructure/perception/test_video_sampler_behavior.py
git commit -m "feat: add clip sampler and vjepa clip scoring boundary"
```

### Task 3: 加入 best-frame 选择和 raw fusion

**Files:**
- Modify: `src/allinone/infrastructure/perception/yolo/detector.py`
- Modify: `src/allinone/application/runtime/build_clip_perception_payload.py`
- Modify: `tests/infrastructure/perception/test_perception_adapter_behavior.py`
- Modify: `tests/application/test_runtime_usecases.py`

**Step 1: Write the failing test**

新增测试，验证：

- 多帧检测结果能选出 `best_frame_index`
- best frame 的 `prediction_rows` 会进入 raw payload
- `vjepa` 字段会保留 `stability_score/alignment_score`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/perception/test_perception_adapter_behavior.py tests/application/test_runtime_usecases.py -v`
Expected: FAIL because best-frame selection is missing.

**Step 3: Write minimal implementation**

- 为 detector 增加 best-frame 选择 helper
- 在 clip use case 中融合 YOLO 与 V-JEPA 输出

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/perception/test_perception_adapter_behavior.py tests/application/test_runtime_usecases.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/perception/yolo/detector.py src/allinone/application/runtime/build_clip_perception_payload.py tests/infrastructure/perception/test_perception_adapter_behavior.py tests/application/test_runtime_usecases.py
git commit -m "feat: add clip fusion and best-frame selection"
```

### Task 4: 增加 `analyze-clip` CLI

**Files:**
- Modify: `src/allinone/interfaces/cli/main.py`
- Modify: `tests/interfaces/cli/test_cli_smoke.py`
- Modify: `experiments/README.md`

**Step 1: Write the failing test**

新增 CLI 测试，验证：

- `analyze-clip` 命令存在
- 能接收 `--clip --yolo-model --vjepa-repo --vjepa-checkpoint --targets --output`
- 能写出 clip 级 raw perception JSON

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -v`
Expected: FAIL because `analyze-clip` command does not exist.

**Step 3: Write minimal implementation**

- 在 CLI parser 中增加 `analyze-clip`
- 调用 clip use case
- 文档补充命令链

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/interfaces/cli/main.py tests/interfaces/cli/test_cli_smoke.py experiments/README.md
git commit -m "feat: add analyze-clip cli command"
```

### Task 5: 跑本地和服务器验证

**Files:**
- Reuse existing files only

**Step 1: Run local targeted tests**

Run:

```bash
python3 -m pytest tests/application/test_runtime_usecases.py tests/infrastructure/perception/test_vjepa_adapter_behavior.py tests/infrastructure/perception/test_video_sampler_behavior.py tests/interfaces/cli/test_cli_smoke.py -v
```

Expected: PASS

**Step 2: Run full local test suite**

Run:

```bash
python3 -m pytest tests -v
```

Expected: PASS

**Step 3: Sync to server and verify**

Run:

```bash
bash ops/remote/sync_to_server.sh
ssh dell@192.168.1.104 'cd /home/dell/workspaces/allinone && . .venv/bin/activate && export PYTHONPATH=/home/dell/workspaces/allinone:/home/dell/workspaces/allinone/src && python3 -m pytest tests -v'
```

Expected: PASS

**Step 4: Run server clip smoke**

基于服务器已有资产：

- repo: `/home/dell/vjepa2`
- checkpoint: `/home/dell/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt`
- YOLO: `/home/dell/yolo11n.pt`

执行：

```bash
python3 -m allinone.interfaces.cli.main analyze-clip \
  --clip <server-clip.mp4> \
  --yolo-model /home/dell/yolo11n.pt \
  --vjepa-repo /home/dell/vjepa2 \
  --vjepa-checkpoint /home/dell/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt \
  --targets meter \
  --output /tmp/allinone-clip-raw.json
python3 -m allinone.interfaces.cli.main build-observation-payload --input /tmp/allinone-clip-raw.json --output /tmp/allinone-clip-payload.json
python3 -m allinone.interfaces.cli.main runtime-observation --input /tmp/allinone-clip-payload.json
```

Expected: raw JSON 生成成功，payload 生成成功，runtime-observation 输出 guidance。

**Step 5: Commit**

```bash
git add .
git commit -m "test: verify vjepa analyze-clip closed loop"
```
