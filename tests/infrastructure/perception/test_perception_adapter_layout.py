from importlib import import_module


def test_perception_adapters_exist():
    modules = [
        "allinone.infrastructure.perception.yolo.detector",
        "allinone.infrastructure.perception.vjepa.encoder",
        "allinone.infrastructure.perception.fusion.observation_builder",
    ]
    for module_name in modules:
        module = import_module(module_name)
        assert module is not None
