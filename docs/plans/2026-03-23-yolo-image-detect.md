# YOLO Single Image Detect Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `allinone` 新增 `detect-image` 命令，把单张图片通过 YOLO 转成 raw perception JSON，并接入现有运行主链。

**Architecture:** 保持现有 `raw perception -> standardized payload -> runtime-observation` 边界不变，只在前面增加“单图检测”入口。`application` 负责编排 raw payload 结构，`infrastructure` 负责真实 YOLO 检测，`interfaces` 负责 CLI 输入输出。

**Tech Stack:** Python 3.14, pytest, pathlib, json, Pillow, Ultralytics YOLO

---

### Task 1: 增加单图 raw perception 用例测试

**Files:**
- Create: `src/allinone/application/runtime/build_raw_perception_payload.py`
- Modify: `tests/application/test_runtime_usecases.py`
- Modify: `tests/application/test_usecase_layout.py`

**Step 1: Write the failing test**

新增测试，验证：

- 输入图片路径、目标标签和假 detector
- 用例返回的 payload 包含 `detections` 和 `vjepa`
- `image_size` 从图片文件读取
- `prediction_rows` 由 detector 结果生成

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/application/test_runtime_usecases.py tests/application/test_usecase_layout.py -v`
Expected: FAIL because use case module does not exist.

**Step 3: Write minimal implementation**

- 新建 `build_raw_perception_payload.py`
- 提供 `build_raw_perception_payload_from_image(...)`
- 用 Pillow 读取图片尺寸
- 调用 detector adapter 的 `predict(...)`
- 输出 raw perception JSON

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/application/test_runtime_usecases.py tests/application/test_usecase_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/application/runtime/build_raw_perception_payload.py tests/application/test_runtime_usecases.py tests/application/test_usecase_layout.py
git commit -m "feat: add raw perception image use case"
```

### Task 2: 增加 `detect-image` CLI 测试

**Files:**
- Modify: `src/allinone/interfaces/cli/main.py`
- Modify: `tests/interfaces/cli/test_cli_smoke.py`

**Step 1: Write the failing test**

新增 CLI 测试，验证：

- `detect-image` 命令存在
- 能接收 `--image --model --targets --output`
- 调用 use case 后写出 raw perception JSON

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -v`
Expected: FAIL because CLI command does not exist.

**Step 3: Write minimal implementation**

- 在 parser 中新增 `detect-image`
- 解析 `targets`
- 调用 raw perception 用例
- 输出 JSON 文件

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/interfaces/cli/test_cli_smoke.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/interfaces/cli/main.py tests/interfaces/cli/test_cli_smoke.py
git commit -m "feat: add detect-image cli command"
```

### Task 3: 补 detector 结构导出和文档样例

**Files:**
- Modify: `src/allinone/infrastructure/perception/yolo/detector.py`
- Modify: `experiments/README.md`
- Create: `experiments/samples/detect_image_command.md`
- Modify: `tests/infrastructure/perception/test_perception_adapter_behavior.py`

**Step 1: Write the failing test**

新增测试，验证 detector 结果可以稳定导出为 `prediction_rows` 样式。

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/perception/test_perception_adapter_behavior.py -v`
Expected: FAIL because export helper does not exist.

**Step 3: Write minimal implementation**

- 为 `DetectionCandidate` 增加导出 helper
- 在文档中增加命令链示例

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/perception/test_perception_adapter_behavior.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/perception/yolo/detector.py experiments/README.md experiments/samples/detect_image_command.md tests/infrastructure/perception/test_perception_adapter_behavior.py
git commit -m "feat: add detector row export helper"
```

### Task 4: 跑闭环验证

**Files:**
- Reuse existing files only

**Step 1: Run targeted tests**

Run:

```bash
python3 -m pytest tests/application/test_runtime_usecases.py tests/interfaces/cli/test_cli_smoke.py tests/infrastructure/perception/test_perception_adapter_behavior.py -v
```

Expected: PASS

**Step 2: Run full test suite**

Run:

```bash
python3 -m pytest tests -v
```

Expected: PASS

**Step 3: Run local smoke chain**

用一张本地测试图跑：

```bash
PYTHONPATH=src python3 -m allinone.interfaces.cli.main detect-image --image <img> --model <yolo.pt> --targets meter --output /tmp/raw.json
PYTHONPATH=src python3 -m allinone.interfaces.cli.main build-observation-payload --input /tmp/raw.json --output /tmp/payload.json
PYTHONPATH=src python3 -m allinone.interfaces.cli.main runtime-observation --input /tmp/payload.json
```

Expected: raw JSON、payload JSON 正常生成，运行主链输出 guidance 结果。

**Step 4: Commit**

```bash
git add .
git commit -m "test: verify yolo single-image closed loop"
```
