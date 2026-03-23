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
