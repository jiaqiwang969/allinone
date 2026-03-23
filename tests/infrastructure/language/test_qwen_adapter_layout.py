from importlib import import_module
from pathlib import Path

from tests._repo import repo_root


def test_qwen_adapter_and_recipe_exist():
    modules = [
        "allinone.infrastructure.language.qwen.client",
        "allinone.infrastructure.language.qwen.prompt_builder",
        "allinone.infrastructure.language.qwen.structured_output",
    ]
    for module_name in modules:
        module = import_module(module_name)
        assert module is not None

    recipe = repo_root() / "configs/model_recipes/qwen35_9b.yaml"
    assert recipe.exists()
