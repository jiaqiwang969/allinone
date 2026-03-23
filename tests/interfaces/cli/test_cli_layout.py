from pathlib import Path


def test_cli_and_remote_scripts_exist():
    root = Path("/Users/jqwang/31-allinone")
    expected = [
        "src/allinone/interfaces/cli/main.py",
        "ops/remote/bootstrap_server.sh",
        "ops/remote/sync_to_server.sh",
        "ops/remote/run_runtime_loop.sh",
    ]
    for rel in expected:
        assert (root / rel).exists(), rel
