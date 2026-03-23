from pathlib import Path


def test_architecture_mapping_docs_and_data_recipe_exist():
    root = Path("/Users/jqwang/31-allinone")
    expected = {
        "docs/architecture/autoresearch-mapping.md": "ExperimentRun",
        "docs/architecture/perception-adapters.md": "YOLO",
        "docs/architecture/migration-map.md": "29-autoresearch",
        "docs/architecture/source-assets.md": "ultralytics",
        "configs/data_recipes/m400_phase1.yaml": "session",
    }

    for relative_path, expected_token in expected.items():
        path = root / relative_path
        assert path.exists(), relative_path
        assert expected_token in path.read_text(encoding="utf-8"), relative_path
