from pathlib import Path

from tests._repo import repo_root


def test_cli_and_remote_scripts_exist():
    root = repo_root()
    expected = [
        "src/allinone/interfaces/cli/main.py",
        "ops/remote/bootstrap_server.sh",
        "ops/remote/sync_to_server.sh",
        "ops/remote/run_runtime_loop.sh",
    ]
    for rel in expected:
        assert (root / rel).exists(), rel
