from pathlib import Path

from tests._repo import repo_root


def test_remote_ops_scripts_contain_real_sync_and_runtime_commands():
    root = repo_root()
    expectations = {
        "ops/remote/bootstrap_server.sh": [
            "python3 -m venv",
            "pip install -e",
            "/home/dell/workspaces/allinone",
            "--exclude '.venv'",
            "--exclude '.worktrees'",
        ],
        "ops/remote/install_qwen_runtime.sh": [
            "/home/dell/workspaces/allinone",
            "https://download.pytorch.org/whl/cu126",
            "torch==2.7.1",
            "torchvision==0.22.1",
            "torchaudio==2.7.1",
            "transformers",
            "accelerate",
            "sentencepiece",
            "safetensors",
        ],
        "ops/remote/sync_to_server.sh": [
            "rsync",
            "dell@192.168.1.104",
            "/home/dell/workspaces/allinone",
            "--exclude '.venv'",
            "--exclude '.worktrees'",
        ],
        "ops/remote/run_runtime_loop.sh": [
            "/home/dell/workspaces/allinone",
            "python3 -m allinone.interfaces.cli.main guidance-smoke",
            "python3 -m allinone.interfaces.cli.main language-smoke",
            "python3 -m allinone.interfaces.cli.main research-smoke",
        ],
    }

    for relative_path, snippets in expectations.items():
        content = (root / relative_path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in content, f"{relative_path}: missing {snippet}"
