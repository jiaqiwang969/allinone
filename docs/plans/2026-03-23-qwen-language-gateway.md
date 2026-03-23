# Qwen 语言网关与常驻服务 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `allinone` 增加统一语言网关与 Qwen 常驻服务，让 CLI、runtime 和 research 都能跨进程复用同一个离线 9B 语言运行时。

**Architecture:** 在 `infrastructure/language/qwen` 内引入统一网关，底层同时支持本地直连和常驻服务调用；在 `interfaces` 中增加 `serve-qwen` 服务入口；在 `ops/remote` 中增加服务器端启动与健康检查脚本。业务层保持对语言能力的抽象依赖，不直接接触 HTTP 或服务配置细节。

**Tech Stack:** Python 3.10+, pytest, standard library `http.server`/`urllib` or minimal HTTP stack already available in repo, pathlib, JSON, existing allinone CLI/runtime modules

---

### Task 1: 定义语言网关配置与服务协议

**Files:**
- Create: `src/allinone/infrastructure/language/qwen/schemas.py`
- Create: `configs/model_recipes/qwen_gateway.yaml`
- Test: `tests/infrastructure/language/test_qwen_gateway_behavior.py`

**Step 1: Write the failing test**

- 写测试验证服务请求与响应结构能被正确序列化
- 写测试验证网关 recipe 能解析出：
  - `mode`
  - `service_url`
  - `service_timeout_seconds`
  - 本地模型参数

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/language/test_qwen_gateway_behavior.py -k schema -v`
Expected: FAIL because schema/config files do not exist yet.

**Step 3: Write minimal implementation**

- 定义轻量 dataclass：
  - `QwenServiceGenerateRequest`
  - `QwenServiceGenerateResponse`
  - `QwenGatewayConfig`
- 增加配置文件 `configs/model_recipes/qwen_gateway.yaml`
- 只实现最小可读写逻辑，不加额外复杂字段

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/language/test_qwen_gateway_behavior.py -k schema -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/language/qwen/schemas.py configs/model_recipes/qwen_gateway.yaml tests/infrastructure/language/test_qwen_gateway_behavior.py
git commit -m "feat: add qwen gateway config and schemas"
```

### Task 2: 增加服务客户端与网关选择逻辑

**Files:**
- Create: `src/allinone/infrastructure/language/qwen/service_client.py`
- Create: `src/allinone/infrastructure/language/qwen/gateway.py`
- Modify: `src/allinone/infrastructure/language/qwen/client.py`
- Test: `tests/infrastructure/language/test_qwen_gateway_behavior.py`

**Step 1: Write the failing test**

- 写测试验证：
  - 服务健康时优先走服务客户端
  - 服务不可用时回退本地 `QwenClient`
  - 服务和本地都不可用时，调用方能收到明确失败
- 写测试验证服务返回文本后仍会执行输出净化

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/infrastructure/language/test_qwen_gateway_behavior.py -k gateway -v`
Expected: FAIL because gateway/service client do not exist yet.

**Step 3: Write minimal implementation**

- 用最简单 HTTP JSON 请求实现 `service_client`
- 在 `gateway.py` 中实现：
  - `mode=auto`
  - `mode=service`
  - `mode=local`
- 复用 `QwenClient` 的文本清洗逻辑，不复制一套不同规则

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/infrastructure/language/test_qwen_gateway_behavior.py -k gateway -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/infrastructure/language/qwen/service_client.py src/allinone/infrastructure/language/qwen/gateway.py src/allinone/infrastructure/language/qwen/client.py tests/infrastructure/language/test_qwen_gateway_behavior.py
git commit -m "feat: add qwen service client and gateway"
```

### Task 3: 增加 Qwen 常驻服务入口

**Files:**
- Create: `src/allinone/interfaces/qwen_service.py`
- Modify: `src/allinone/interfaces/cli/main.py`
- Test: `tests/interfaces/cli/test_qwen_service_cli.py`

**Step 1: Write the failing test**

- 写测试验证 `serve-qwen` 子命令存在
- 写测试验证服务进程暴露：
  - `GET /health`
  - `POST /generate`
- 写测试验证 `POST /generate` 返回的文本是净化后的结果

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/interfaces/cli/test_qwen_service_cli.py -v`
Expected: FAIL because service entrypoint and CLI command do not exist yet.

**Step 3: Write minimal implementation**

- 选用最小 HTTP 服务实现
- 启动时只初始化一次本地 `QwenClient`
- `generate` 只接受当前闭环所需字段
- `main.py` 增加 `serve-qwen`

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/interfaces/cli/test_qwen_service_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/interfaces/qwen_service.py src/allinone/interfaces/cli/main.py tests/interfaces/cli/test_qwen_service_cli.py
git commit -m "feat: add qwen service entrypoint"
```

### Task 4: 让 CLI runtime 路径切到语言网关

**Files:**
- Modify: `src/allinone/application/runtime/run_runtime_observation.py`
- Modify: `src/allinone/interfaces/cli/main.py`
- Test: `tests/interfaces/cli/test_language_smoke.py`
- Test: `tests/application/test_runtime_usecases.py`

**Step 1: Write the failing test**

- 写测试验证 `language-smoke` 优先走语言网关
- 写测试验证 `_CliRuntimeTextGenerator` 在服务可用时走 `service`
- 写测试验证服务不可用时仍可退回 `local` / `mock`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/interfaces/cli/test_language_smoke.py tests/application/test_runtime_usecases.py -k gateway -v`
Expected: FAIL because current runtime path still直连 `QwenClient`.

**Step 3: Write minimal implementation**

- 在 CLI 和 runtime 中接入统一网关
- 保持返回结构、提示词构造与解析逻辑不变
- 不改变 domain / application 的业务语义

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/interfaces/cli/test_language_smoke.py tests/application/test_runtime_usecases.py -k gateway -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/allinone/application/runtime/run_runtime_observation.py src/allinone/interfaces/cli/main.py tests/interfaces/cli/test_language_smoke.py tests/application/test_runtime_usecases.py
git commit -m "feat: route runtime language calls through qwen gateway"
```

### Task 5: 增加远端运维脚本

**Files:**
- Create: `ops/remote/start_qwen_service.sh`
- Create: `ops/remote/check_qwen_service.sh`
- Modify: `ops/remote/bootstrap_server.sh`
- Test: `tests/interfaces/cli/test_remote_ops_scripts_behavior.py`

**Step 1: Write the failing test**

- 写测试验证新脚本存在
- 写测试验证脚本包含：
  - 启动 `serve-qwen`
  - 健康检查 `/health`
  - 使用服务器上的固定模型路径或 recipe

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/interfaces/cli/test_remote_ops_scripts_behavior.py -k qwen -v`
Expected: FAIL because scripts do not exist yet.

**Step 3: Write minimal implementation**

- 添加远端启动脚本
- 添加健康检查脚本
- 在 bootstrap 文档或脚本中补充调用方式

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/interfaces/cli/test_remote_ops_scripts_behavior.py -k qwen -v`
Expected: PASS

**Step 5: Commit**

```bash
git add ops/remote/start_qwen_service.sh ops/remote/check_qwen_service.sh ops/remote/bootstrap_server.sh tests/interfaces/cli/test_remote_ops_scripts_behavior.py
git commit -m "feat: add remote qwen service ops scripts"
```

### Task 6: 完整验证与远端真实回归

**Files:**
- Reuse existing files only

**Step 1: Run targeted tests**

Run:

```bash
python3 -m pytest tests/infrastructure/language/test_qwen_gateway_behavior.py -v
python3 -m pytest tests/interfaces/cli/test_qwen_service_cli.py -v
python3 -m pytest tests/interfaces/cli/test_language_smoke.py tests/application/test_runtime_usecases.py -v
```

Expected: PASS

**Step 2: Run full suite**

Run: `python3 -m pytest tests -v`
Expected: PASS

**Step 3: Sync to server**

Run:

```bash
RSYNC_RSH="sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no" bash ops/remote/sync_to_server.sh
```

Expected: server workspace updated successfully.

**Step 4: Start Qwen service on server**

Run:

```bash
sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no dell@192.168.1.104 'cd /home/dell/workspaces/allinone && bash ops/remote/start_qwen_service.sh'
sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no dell@192.168.1.104 'cd /home/dell/workspaces/allinone && bash ops/remote/check_qwen_service.sh'
```

Expected: service becomes healthy.

**Step 5: Run real server sanity checks**

Run:

```bash
sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no dell@192.168.1.104 'cd /home/dell/workspaces/allinone && . .venv/bin/activate && export PYTHONPATH=/home/dell/workspaces/allinone:/home/dell/workspaces/allinone/src && python3 -m allinone.interfaces.cli.main language-smoke'
```

Expected:

- `source=qwen`
- 不再每次命令都重新装载 427 shards

**Step 6: Run one real research step**

Run one small replay experiment on server, for example:

```bash
sshpass -p 'root@123' ssh -o StrictHostKeyChecking=no dell@192.168.1.104 'cd /home/dell/workspaces/allinone && . .venv/bin/activate && export PYTHONPATH=/home/dell/workspaces/allinone:/home/dell/workspaces/allinone/src && python3 -m allinone.interfaces.cli.main run-research-step --experiment-id exp-loop-service-001 --hypothesis "reuse qwen service across research loop" --target-metric guidance_success_rate --manifest /home/dell/workspaces/allinone/experiments/generated/qwen-reuse-check/person_boundary_replay/manifest.jsonl --base-policy /home/dell/workspaces/allinone/configs/runtime_policies/m400_default.json --candidate-count 3 --run-root /home/dell/workspaces/allinone/experiments/research/exp-loop-service-001 --output /home/dell/workspaces/allinone/experiments/research/exp-loop-service-001/summary.json --yolo-model /home/dell/yolo11n.pt --vjepa-repo /home/dell/vjepa2 --vjepa-checkpoint /home/dell/vjepa2/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt --sample-frames 4'
```

Expected:

- experiment completes
- summary and judgement are written
- language path uses the service-backed gateway
