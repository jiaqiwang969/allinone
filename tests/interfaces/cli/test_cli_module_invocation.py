import os
import subprocess
import sys
from pathlib import Path


def test_cli_module_invocation_runs_guidance_smoke():
    root = Path("/Users/jqwang/31-allinone")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [sys.executable, "-m", "allinone.interfaces.cli.main", "guidance-smoke"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "guidance_action=left" in result.stdout
