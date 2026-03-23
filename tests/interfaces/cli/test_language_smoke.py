from pathlib import Path

from allinone.interfaces.cli import main as cli_main


def test_language_smoke_command_falls_back_to_mock_output(capsys, monkeypatch):
    monkeypatch.setattr(
        cli_main,
        "_resolve_qwen_recipe_path",
        lambda: Path("/tmp/does-not-exist-qwen-recipe.yaml"),
        raising=False,
    )

    exit_code = cli_main.main(["language-smoke"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "language_action=left" in captured.out
    assert "confidence=0.82" in captured.out
    assert "source=mock" in captured.out


def test_language_smoke_command_uses_qwen_gateway_when_runtime_is_available(
    capsys, monkeypatch, tmp_path
):
    recipe = tmp_path / "qwen_gateway.yaml"
    recipe.write_text("model_id: test\nruntime_path: /tmp/qwen\n", encoding="utf-8")

    class FakeGateway:
        def __init__(self, config) -> None:
            self.config = config

        def generate_text(self, prompt: str):
            assert "suggested_action" in prompt
            return (
                """{
                "operator_message": "请向左移动，让仪表回到画面中央。",
                "suggested_action": "left",
                "confidence": 0.93,
                "evidence_focus": "确保整个表盘完整可见"
            }""",
                "service",
            )

    class FakeGatewayConfig:
        @staticmethod
        def from_recipe(path):
            assert path == recipe
            return object()

    monkeypatch.setattr(
        cli_main,
        "_resolve_qwen_gateway_recipe_path",
        lambda: recipe,
        raising=False,
    )
    monkeypatch.setattr(
        cli_main,
        "QwenGatewayConfig",
        FakeGatewayConfig,
        raising=False,
    )
    monkeypatch.setattr(
        cli_main,
        "QwenGateway",
        FakeGateway,
        raising=False,
    )

    exit_code = cli_main.main(["language-smoke"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "language_action=left" in captured.out
    assert "confidence=0.93" in captured.out
    assert "source=service" in captured.out


def test_cli_runtime_text_generator_reuses_qwen_gateway(tmp_path, monkeypatch):
    recipe = tmp_path / "qwen_gateway.yaml"
    recipe.write_text("model_id: test\nruntime_path: /tmp/qwen\n", encoding="utf-8")
    calls = {"from_recipe": 0, "gateway_init": 0, "generate_text": 0}

    class FakeGateway:
        def __init__(self, config) -> None:
            calls["gateway_init"] += 1
            self.config = config

        def generate_text(self, prompt: str):
            calls["generate_text"] += 1
            return (
                """{
                "operator_message": "请向左移动，让仪表回到画面中央。",
                "suggested_action": "left",
                "confidence": 0.93,
                "evidence_focus": "确保整个表盘完整可见"
            }""",
                "service",
            )

    class FakeGatewayConfig:
        @staticmethod
        def from_recipe(path):
            calls["from_recipe"] += 1
            assert path == recipe
            return object()

    monkeypatch.setattr(
        cli_main,
        "_resolve_qwen_gateway_recipe_path",
        lambda: recipe,
        raising=False,
    )
    monkeypatch.setattr(
        cli_main,
        "QwenGatewayConfig",
        FakeGatewayConfig,
        raising=False,
    )
    monkeypatch.setattr(
        cli_main,
        "QwenGateway",
        FakeGateway,
        raising=False,
    )

    generator = cli_main._CliRuntimeTextGenerator()

    assert generator.generate("prompt-1")[1] == "service"
    assert generator.generate("prompt-2")[1] == "service"
    assert calls == {
        "from_recipe": 1,
        "gateway_init": 1,
        "generate_text": 2,
    }
