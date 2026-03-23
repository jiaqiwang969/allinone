# Allinone Repository Bootstrap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `/Users/jqwang/31-allinone` 创建新的 `allinone` 主仓，并按 DDD 架构完成第一阶段骨架、迁移入口和主闭环占位实现。

**Architecture:** 以“远程工业作业会话闭环”为核心域，建立 `domain / application / infrastructure / interfaces / bootstrap` 五层结构。第一阶段只承载 `session -> guidance -> evidence -> research` 主链，把 `YOLO / V-JEPA / autoresearch / Qwen` 通过 adapter 方式挂入，而不是把旧目录原样搬迁。

**Tech Stack:** Python 3.10+, pytest, dataclasses/pydantic（按需）, PyTorch, transformers, Ultralytics, Hugging Face, shell/rsync/git

---

### Task 1: 初始化 `31-allinone` 仓库骨架

**Files:**
- Create: `/Users/jqwang/31-allinone/README.md`
- Create: `/Users/jqwang/31-allinone/pyproject.toml`
- Create: `/Users/jqwang/31-allinone/AGENTS.md`
- Create: `/Users/jqwang/31-allinone/src/allinone/__init__.py`
- Create: `/Users/jqwang/31-allinone/docs/architecture/README.md`
- Create: `/Users/jqwang/31-allinone/docs/plans/README.md`

**Step 1: Write the failing test**

Create `/Users/jqwang/31-allinone/tests/test_repo_layout.py`:

```python
from pathlib import Path


def test_repo_layout_has_core_directories():
    root = Path("/Users/jqwang/31-allinone")
    for rel in [
        "src/allinone",
        "docs/architecture",
        "docs/plans",
        "configs",
        "experiments",
        "ops",
    ]:
        assert (root / rel).exists(), rel
```

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/test_repo_layout.py -v`
Expected: FAIL because repository skeleton does not exist.

**Step 3: Write minimal implementation**

- Create the directory tree
- Add minimal `README.md`, `pyproject.toml`, `src/allinone/__init__.py`, `AGENTS.md`
- Add empty marker files where needed

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/test_repo_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "chore: initialize allinone repository skeleton"
```

### Task 2: 建立 DDD 顶层包边界

**Files:**
- Create: `/Users/jqwang/31-allinone/src/allinone/domain/__init__.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/application/__init__.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/infrastructure/__init__.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/interfaces/__init__.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/bootstrap/__init__.py`
- Test: `/Users/jqwang/31-allinone/tests/test_package_imports.py`

**Step 1: Write the failing test**

```python
def test_core_packages_import():
    import allinone.domain
    import allinone.application
    import allinone.infrastructure
    import allinone.interfaces
    import allinone.bootstrap
```

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/test_package_imports.py -v`
Expected: FAIL with import error.

**Step 3: Write minimal implementation**

- Create the five top-level packages
- Ensure `pyproject.toml` makes `src` layout importable in tests

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/test_package_imports.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "chore: add core ddd package boundaries"
```

### Task 3: 建立核心子域骨架

**Files:**
- Create: `/Users/jqwang/31-allinone/src/allinone/domain/session/{entities.py,value_objects.py,commands.py,events.py,services.py,policies.py,repositories.py,errors.py}`
- Create: `/Users/jqwang/31-allinone/src/allinone/domain/guidance/{entities.py,value_objects.py,commands.py,events.py,services.py,policies.py,repositories.py,errors.py}`
- Create: `/Users/jqwang/31-allinone/src/allinone/domain/evidence/{entities.py,value_objects.py,commands.py,events.py,services.py,policies.py,repositories.py,errors.py}`
- Create: `/Users/jqwang/31-allinone/src/allinone/domain/perception/{entities.py,value_objects.py,commands.py,events.py,services.py,policies.py,repositories.py,errors.py}`
- Create: `/Users/jqwang/31-allinone/src/allinone/domain/research/{entities.py,value_objects.py,commands.py,events.py,services.py,policies.py,repositories.py,errors.py}`
- Create: `/Users/jqwang/31-allinone/src/allinone/domain/shared/{entities.py,value_objects.py,events.py,errors.py}`
- Test: `/Users/jqwang/31-allinone/tests/domain/test_domain_layout.py`

**Step 1: Write the failing test**

Write assertions that each required module file exists.

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/domain/test_domain_layout.py -v`
Expected: FAIL because subdomain files are missing.

**Step 3: Write minimal implementation**

- Create the subpackages and files
- Add docstrings stating each subdomain responsibility

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/domain/test_domain_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "chore: scaffold core domain subpackages"
```

### Task 4: 定义共享值对象和标识

**Files:**
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/shared/value_objects.py`
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/shared/errors.py`
- Test: `/Users/jqwang/31-allinone/tests/domain/shared/test_value_objects.py`

**Step 1: Write the failing test**

Write tests for:

- `SessionId`
- `StageType`
- `PromptAction`
- `BoundingBox`
- `CenterOffset`

Include validation examples, such as invalid `PromptAction` or malformed box coordinates.

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/domain/shared/test_value_objects.py -v`
Expected: FAIL because the value objects do not exist yet.

**Step 3: Write minimal implementation**

- Implement immutable value objects
- Add domain validation errors

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/domain/shared/test_value_objects.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "feat: add shared domain value objects"
```

### Task 5: 定义 `WorkSession` 聚合与基础不变量

**Files:**
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/session/entities.py`
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/session/events.py`
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/session/commands.py`
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/session/errors.py`
- Test: `/Users/jqwang/31-allinone/tests/domain/session/test_work_session.py`

**Step 1: Write the failing test**

Cover:

- session must require `task_type`
- session starts closed -> opened explicitly
- only one current stage can exist at a time

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/domain/session/test_work_session.py -v`
Expected: FAIL because `WorkSession` aggregate is missing.

**Step 3: Write minimal implementation**

- Add `WorkSession`
- Add `OpenWorkSession`, `AttachTaskPlan`, `StartStage`, `CloseWorkSession`
- Emit corresponding domain events

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/domain/session/test_work_session.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "feat: add work session aggregate"
```

### Task 6: 定义 `PerceptionObservation` 和 `GuidanceDecision`

**Files:**
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/perception/entities.py`
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/guidance/entities.py`
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/guidance/policies.py`
- Test: `/Users/jqwang/31-allinone/tests/domain/guidance/test_guidance_policy.py`

**Step 1: Write the failing test**

Test scenarios:

- centered and ready observation -> `hold_still` or `start_recording`
- target shifted right -> `left`
- target too large -> `backward`

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/domain/guidance/test_guidance_policy.py -v`
Expected: FAIL because observation and policy objects are missing.

**Step 3: Write minimal implementation**

- Define `PerceptionObservation`
- Define `GuidanceDecision`
- Implement minimal `GuidancePolicyService`

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/domain/guidance/test_guidance_policy.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "feat: add observation and guidance domain objects"
```

### Task 7: 定义 `EvidenceBundle` 与验收判定

**Files:**
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/evidence/entities.py`
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/evidence/services.py`
- Modify: `/Users/jqwang/31-allinone/src/allinone/domain/evidence/policies.py`
- Test: `/Users/jqwang/31-allinone/tests/domain/evidence/test_evidence_bundle.py`

**Step 1: Write the failing test**

Cover:

- bundle initially incomplete
- after required screenshot/clip added, bundle becomes acceptable
- invalid evidence type rejected

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/domain/evidence/test_evidence_bundle.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Add `EvidenceItem` and `EvidenceBundle`
- Add acceptance rules and assessment service

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/domain/evidence/test_evidence_bundle.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "feat: add evidence bundle domain model"
```

### Task 8: 建立应用层用例骨架

**Files:**
- Create: `/Users/jqwang/31-allinone/src/allinone/application/runtime/ingest_observation_window.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/application/runtime/request_guidance_decision.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/application/runtime/capture_evidence.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/application/session/open_session.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/application/research/register_experiment.py`
- Test: `/Users/jqwang/31-allinone/tests/application/test_usecase_layout.py`

**Step 1: Write the failing test**

Assert required use case modules exist and expose one public callable each.

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/application/test_usecase_layout.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Add use case modules
- Add placeholder handlers with docstrings and typed signatures

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/application/test_usecase_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "chore: scaffold application use cases"
```

### Task 9: 接入现有 `autoresearch` 逻辑的 research adapter

**Files:**
- Create: `/Users/jqwang/31-allinone/src/allinone/infrastructure/research/autoresearch/__init__.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/infrastructure/research/autoresearch/replay_adapter.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/infrastructure/research/autoresearch/judge_adapter.py`
- Create: `/Users/jqwang/31-allinone/docs/architecture/autoresearch-mapping.md`
- Test: `/Users/jqwang/31-allinone/tests/infrastructure/research/test_autoresearch_adapter_layout.py`

**Step 1: Write the failing test**

Assert adapter modules exist and expose adapter class names.

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/infrastructure/research/test_autoresearch_adapter_layout.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Create research adapter package
- Document which old modules map into new adapter responsibilities

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/infrastructure/research/test_autoresearch_adapter_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "chore: add autoresearch adapter skeleton"
```

### Task 10: 建立 `YOLO` 与 `V-JEPA` 感知适配层

**Files:**
- Create: `/Users/jqwang/31-allinone/src/allinone/infrastructure/perception/yolo/detector.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/infrastructure/perception/vjepa/encoder.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/infrastructure/perception/fusion/observation_builder.py`
- Create: `/Users/jqwang/31-allinone/docs/architecture/perception-adapters.md`
- Test: `/Users/jqwang/31-allinone/tests/infrastructure/perception/test_perception_adapter_layout.py`

**Step 1: Write the failing test**

Assert the three adapter modules exist and each returns domain-facing objects rather than raw upstream result types.

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/infrastructure/perception/test_perception_adapter_layout.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Create detector/encoder/fusion adapter placeholders
- Document raw upstream type boundaries in docstrings

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/infrastructure/perception/test_perception_adapter_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "chore: add perception adapter skeleton"
```

### Task 11: 建立 `Qwen3.5-9B` 语言层适配骨架

**Files:**
- Create: `/Users/jqwang/31-allinone/src/allinone/infrastructure/language/qwen/client.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/infrastructure/language/qwen/prompt_builder.py`
- Create: `/Users/jqwang/31-allinone/src/allinone/infrastructure/language/qwen/structured_output.py`
- Create: `/Users/jqwang/31-allinone/configs/model_recipes/qwen35_9b.yaml`
- Test: `/Users/jqwang/31-allinone/tests/infrastructure/language/test_qwen_adapter_layout.py`

**Step 1: Write the failing test**

Assert the Qwen adapter package and model recipe file exist.

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/infrastructure/language/test_qwen_adapter_layout.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Create Qwen adapter skeleton
- Add model recipe pointing to the remote model path convention

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/infrastructure/language/test_qwen_adapter_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "chore: add qwen language adapter skeleton"
```

### Task 12: 建立运行入口和远端部署脚本

**Files:**
- Create: `/Users/jqwang/31-allinone/src/allinone/interfaces/cli/main.py`
- Create: `/Users/jqwang/31-allinone/ops/remote/bootstrap_server.sh`
- Create: `/Users/jqwang/31-allinone/ops/remote/sync_to_server.sh`
- Create: `/Users/jqwang/31-allinone/ops/remote/run_runtime_loop.sh`
- Test: `/Users/jqwang/31-allinone/tests/interfaces/cli/test_cli_layout.py`

**Step 1: Write the failing test**

Assert CLI entry module and remote ops scripts exist and are executable.

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/interfaces/cli/test_cli_layout.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Add CLI placeholder
- Add shell scripts with safe stubs and comments

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/interfaces/cli/test_cli_layout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "chore: add cli and remote ops skeleton"
```

### Task 13: 建立迁移清单与资产映射

**Files:**
- Create: `/Users/jqwang/31-allinone/docs/architecture/migration-map.md`
- Create: `/Users/jqwang/31-allinone/docs/architecture/source-assets.md`
- Create: `/Users/jqwang/31-allinone/configs/data_recipes/m400_phase1.yaml`
- Test: `/Users/jqwang/31-allinone/tests/docs/test_migration_docs_exist.py`

**Step 1: Write the failing test**

Assert migration docs and phase-1 data recipe exist.

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/docs/test_migration_docs_exist.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Map old directories to new bounded contexts
- Write phase-1 recipe for M400 guidance/evidence loop

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/docs/test_migration_docs_exist.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "docs: add migration map and phase1 asset recipes"
```

### Task 14: 初始化 git 仓库并准备远端同步

**Files:**
- Modify: `/Users/jqwang/31-allinone/README.md`
- Modify: `/Users/jqwang/31-allinone/.gitignore`
- Test: `/Users/jqwang/31-allinone/tests/test_git_files.py`

**Step 1: Write the failing test**

Assert `.gitignore` excludes model weights, datasets, cache, checkpoints, logs, and Hugging Face cache.

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/test_git_files.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Add `.gitignore`
- Update `README.md` with repository bootstrap and push notes
- Initialize git
- Add remote `git@github.com:jiaqiwang969/allinone.git`

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/test_git_files.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "chore: initialize git metadata and repository hygiene"
```

### Task 15: 同步新仓到远端服务器并做最小冒烟验证

**Files:**
- Modify: `/Users/jqwang/31-allinone/ops/remote/sync_to_server.sh`
- Modify: `/Users/jqwang/31-allinone/ops/remote/bootstrap_server.sh`
- Test: `/Users/jqwang/31-allinone/tests/ops/test_remote_scripts.py`

**Step 1: Write the failing test**

Assert remote scripts include:

- target host `dell@192.168.1.104`
- target path convention
- rsync-safe excludes

**Step 2: Run test to verify it fails**

Run: `pytest /Users/jqwang/31-allinone/tests/ops/test_remote_scripts.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

- Add sync/bootstrap script content
- Document remote model path usage, including `Qwen3.5-9B` server path

**Step 4: Run test to verify it passes**

Run: `pytest /Users/jqwang/31-allinone/tests/ops/test_remote_scripts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /Users/jqwang/31-allinone
git add .
git commit -m "ops: add remote sync and bootstrap scripts"
```

## Verification Sweep

After completing all tasks, run:

```bash
cd /Users/jqwang/31-allinone
pytest tests -v
```

Expected:

- Core package import tests pass
- Domain layout and aggregate tests pass
- Adapter skeleton tests pass
- Remote ops and docs existence tests pass

## Migration Cutover Checklist

Before switching primary development to `31-allinone`, confirm:

1. `31-allinone` has a clean git history and remote configured
2. DDD package boundaries exist and are test-covered
3. `session -> guidance -> evidence -> research` main chain has placeholders or migrated implementations
4. Old asset mapping is documented
5. Remote server sync path is defined
6. Model/data/cache ignore rules are active

Plan complete and saved to `docs/plans/2026-03-23-allinone-implementation.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
