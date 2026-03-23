from importlib import import_module


def test_guidance_infrastructure_modules_exist():
    modules = [
        "allinone.infrastructure.guidance.policy_recipe",
    ]
    for module_name in modules:
        module = import_module(module_name)
        assert module is not None
