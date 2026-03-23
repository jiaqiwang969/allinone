import json

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


def test_runtime_observation_command_prints_guidance_and_language(
    capsys, monkeypatch, tmp_path
):
    payload = tmp_path / "observation.json"
    payload.write_text(
        json.dumps(
            {
                "prediction_rows": [
                    {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]}
                ],
                "image_size": [1000, 1000],
                "target_labels": ["meter"],
                "visibility_score": 0.85,
                "readable_ratio": 0.8,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALLINONE_QWEN_RECIPE", str(tmp_path / "missing-qwen.yaml"))

    exit_code = main(["runtime-observation", "--input", str(payload)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "guidance_action=left" in captured.out
    assert "reason=target_shifted_right" in captured.out
    assert "language_action=left" in captured.out
    assert "source=mock" in captured.out
