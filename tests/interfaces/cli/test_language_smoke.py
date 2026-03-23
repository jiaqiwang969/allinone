from allinone.interfaces.cli.main import main


def test_language_smoke_command_prints_structured_result(capsys):
    exit_code = main(["language-smoke"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "language_action=left" in captured.out
    assert "confidence=0.82" in captured.out
