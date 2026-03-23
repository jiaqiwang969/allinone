from pathlib import Path

from tests._repo import repo_root


def test_repo_layout_has_core_directories():
    root = repo_root()
    for rel in [
        "src/allinone",
        "docs/architecture",
        "docs/plans",
        "configs",
        "experiments",
        "ops",
    ]:
        assert (root / rel).exists(), rel
