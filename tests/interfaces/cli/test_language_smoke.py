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


def test_language_smoke_command_uses_qwen_client_when_runtime_is_available(
    capsys, monkeypatch, tmp_path
):
    recipe = tmp_path / "qwen35_9b.yaml"
    recipe.write_text("model_id: test\nruntime_path: /tmp/qwen\n", encoding="utf-8")

    class FakeClient:
        def is_runtime_available(self) -> bool:
            return True

        def generate_text(self, prompt: str) -> str:
            assert "suggested_action" in prompt
            return """{
                "operator_message": "请向左移动，让仪表回到画面中央。",
                "suggested_action": "left",
                "confidence": 0.93,
                "evidence_focus": "确保整个表盘完整可见"
            }"""

    monkeypatch.setattr(cli_main, "_resolve_qwen_recipe_path", lambda: recipe, raising=False)
    monkeypatch.setattr(
        cli_main.QwenClient,
        "from_recipe",
        lambda _: FakeClient(),
    )

    exit_code = cli_main.main(["language-smoke"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "language_action=left" in captured.out
    assert "confidence=0.93" in captured.out
    assert "source=qwen" in captured.out
