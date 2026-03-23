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


def test_build_observation_payload_command_writes_standard_payload(tmp_path):
    raw_input = tmp_path / "raw-perception.json"
    output = tmp_path / "payload.json"
    raw_input.write_text(
        json.dumps(
            {
                "detections": {
                    "prediction_rows": [
                        {
                            "label": "meter",
                            "confidence": 0.91,
                            "xyxy": [600, 200, 900, 800],
                        }
                    ],
                    "image_size": [1000, 1000],
                    "target_labels": ["meter"],
                },
                "vjepa": {
                    "visibility_score": 0.85,
                    "readable_ratio": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "build-observation-payload",
            "--input",
            str(raw_input),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload == {
        "prediction_rows": [
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]}
        ],
        "image_size": [1000, 1000],
        "target_labels": ["meter"],
        "visibility_score": 0.85,
        "readable_ratio": 0.8,
    }


def test_detect_image_command_writes_raw_perception_payload(tmp_path, monkeypatch):
    image_path = tmp_path / "frame.png"
    output = tmp_path / "raw.json"
    image_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "allinone.interfaces.cli.main.build_raw_perception_payload_from_image",
        lambda **kwargs: {
            "detections": {
                "prediction_rows": [
                    {
                        "label": "meter",
                        "confidence": 0.91,
                        "xyxy": [600, 200, 900, 800],
                    }
                ],
                "image_size": [1000, 1000],
                "target_labels": ["meter"],
            },
            "vjepa": {
                "visibility_score": 1.0,
                "readable_ratio": 1.0,
            },
        },
    )

    exit_code = main(
        [
            "detect-image",
            "--image",
            str(image_path),
            "--model",
            "mock-yolo.pt",
            "--targets",
            "meter",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["detections"]["target_labels"] == ["meter"]
    assert payload["vjepa"]["visibility_score"] == 1.0


def test_analyze_clip_command_writes_clip_raw_perception_payload(
    tmp_path, monkeypatch
):
    clip_path = tmp_path / "clip.mp4"
    output = tmp_path / "clip-raw.json"
    clip_path.write_text("placeholder", encoding="utf-8")

    def fake_build_raw_perception_payload_from_clip(**kwargs):
        assert kwargs["clip_path"] == str(clip_path)
        assert kwargs["target_labels"] == ("meter",)
        assert kwargs["sampler"].frame_count == 8
        assert kwargs["detector"].model_path == "mock-yolo.pt"
        assert kwargs["clip_scorer"].repo_path == "/models/vjepa2"
        assert kwargs["clip_scorer"].checkpoint_path == "/models/vjepa2/model.pt"
        return {
            "detections": {
                "prediction_rows": [
                    {
                        "label": "meter",
                        "confidence": 0.97,
                        "xyxy": [300, 90, 1000, 650],
                    }
                ],
                "image_size": [1280, 720],
                "target_labels": ["meter"],
                "best_frame_index": 2,
            },
            "vjepa": {
                "visibility_score": 0.88,
                "readable_ratio": 0.81,
                "stability_score": 0.92,
                "alignment_score": 0.87,
            },
        }

    monkeypatch.setattr(
        "allinone.interfaces.cli.main.build_raw_perception_payload_from_clip",
        fake_build_raw_perception_payload_from_clip,
    )

    exit_code = main(
        [
            "analyze-clip",
            "--clip",
            str(clip_path),
            "--yolo-model",
            "mock-yolo.pt",
            "--vjepa-repo",
            "/models/vjepa2",
            "--vjepa-checkpoint",
            "/models/vjepa2/model.pt",
            "--targets",
            "meter",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["detections"]["best_frame_index"] == 2
    assert payload["vjepa"]["alignment_score"] == 0.87
