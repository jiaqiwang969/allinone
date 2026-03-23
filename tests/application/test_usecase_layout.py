from importlib import import_module


def test_application_usecase_modules_exist():
    modules = [
        "allinone.application.runtime.build_raw_perception_payload",
        "allinone.application.runtime.build_observation_payload",
        "allinone.application.runtime.ingest_observation_window",
        "allinone.application.runtime.request_guidance_decision",
        "allinone.application.runtime.capture_evidence",
        "allinone.application.session.open_session",
        "allinone.application.research.register_experiment",
    ]
    for module_name in modules:
        module = import_module(module_name)
        assert module is not None
