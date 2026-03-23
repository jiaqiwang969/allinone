from allinone.interfaces.cli.main import main


def test_guidance_smoke_command_prints_decision(capsys):
    exit_code = main(["guidance-smoke"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "guidance_action=left" in captured.out


def test_research_smoke_command_prints_payload(capsys):
    exit_code = main(["research-smoke"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "experiment_id=exp-smoke-001" in captured.out
    assert "candidate_names=baseline,candidate-a" in captured.out
