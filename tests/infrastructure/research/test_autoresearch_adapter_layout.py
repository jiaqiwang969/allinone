from importlib import import_module


def test_autoresearch_adapters_exist():
    modules = [
        "allinone.infrastructure.research.autoresearch.replay_adapter",
        "allinone.infrastructure.research.autoresearch.judge_adapter",
        "allinone.infrastructure.research.autoresearch.policy_candidate_proposer",
        "allinone.infrastructure.research.autoresearch.rule_based_judge",
        "allinone.infrastructure.research.autoresearch.run_writer",
    ]
    for module_name in modules:
        module = import_module(module_name)
        assert module is not None
