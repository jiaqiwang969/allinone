from pathlib import Path


def test_remote_ops_scripts_contain_real_sync_and_runtime_commands():
    root = Path("/Users/jqwang/31-allinone")
    expectations = {
        "ops/remote/bootstrap_server.sh": ["python3 -m venv", "pip install -e"],
        "ops/remote/sync_to_server.sh": ["rsync", "dell@192.168.1.104"],
        "ops/remote/run_runtime_loop.sh": [
            "python3 -m allinone.interfaces.cli.main guidance-smoke",
            "python3 -m allinone.interfaces.cli.main research-smoke",
        ],
    }

    for relative_path, snippets in expectations.items():
        content = (root / relative_path).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in content, f"{relative_path}: missing {snippet}"
