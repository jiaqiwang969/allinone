import json

from PIL import Image

from allinone.application.runtime.build_clip_perception_payload import (
    build_raw_perception_payload_from_clip,
)
from allinone.application.runtime.build_raw_perception_payload import (
    build_raw_perception_payload_from_image,
)
from allinone.application.runtime.build_observation_payload import (
    build_observation_payload,
)
from allinone.application.runtime.capture_evidence import capture_evidence
from allinone.application.runtime.run_runtime_observation import (
    run_runtime_observation,
)
from allinone.application.runtime.request_guidance_decision import (
    request_guidance_decision,
)
from allinone.application.session.open_session import open_session
from allinone.domain.evidence.entities import EvidenceBundle, EvidenceItem
from allinone.domain.evidence.policies import EvidenceRequirementPolicy
from allinone.domain.guidance.entities import GuidanceDecision
from allinone.domain.guidance.services import GuidanceThresholds
from allinone.domain.perception.entities import PerceptionObservation
from allinone.domain.session.entities import WorkSession
from allinone.infrastructure.guidance.policy_recipe import RuntimePolicyRecipeStore
from allinone.domain.shared.value_objects import BoundingBox, CenterOffset, SessionId, StageType
from allinone.infrastructure.perception.video.sampler import SampledClip
from allinone.infrastructure.perception.vjepa.encoder import ClipQualitySignal
from allinone.infrastructure.perception.yolo.detector import DetectionCandidate


def test_open_session_usecase_returns_open_work_session():
    session = open_session(session_id="session-001", task_type="quality_inspection")

    assert isinstance(session, WorkSession)
    assert session.status == "open"
    assert session.task_type == "quality_inspection"


def test_request_guidance_decision_usecase_returns_domain_decision():
    observation = PerceptionObservation(
        visibility_score=0.7,
        readable_ratio=0.8,
        fill_ratio=0.5,
        center_offset=CenterOffset(dx=0.25, dy=0.0),
        roi=BoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9),
    )

    decision = request_guidance_decision(observation)

    assert isinstance(decision, GuidanceDecision)
    assert decision.action.value == "left"


def test_request_guidance_decision_can_use_custom_thresholds():
    observation = PerceptionObservation(
        visibility_score=0.7,
        readable_ratio=0.8,
        fill_ratio=0.5,
        center_offset=CenterOffset(dx=0.25, dy=0.0),
        roi=BoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9),
    )

    decision = request_guidance_decision(
        observation,
        guidance_thresholds=GuidanceThresholds(directional_offset_min=0.3),
    )

    assert isinstance(decision, GuidanceDecision)
    assert decision.action.value == "hold_still"


def test_runtime_observation_can_load_guidance_thresholds_from_policy_recipe(tmp_path):
    class FakeTextGenerator:
        def generate(self, prompt: str):
            assert "suggested_action=hold_still" in prompt
            return (
                '{"operator_message":"保持稳定","suggested_action":"hold_still","confidence":0.61,"evidence_focus":"保持目标稳定"}',
                "fake-qwen",
            )

    recipe_path = tmp_path / "policy.json"
    recipe_path.write_text(
        json.dumps(
            {
                "guidance_thresholds": {
                    "centered_offset_max": 0.09,
                    "directional_offset_min": 0.3,
                    "ready_fill_ratio_max": 0.85,
                }
            }
        ),
        encoding="utf-8",
    )
    thresholds = RuntimePolicyRecipeStore().load_guidance_thresholds(recipe_path)

    result = run_runtime_observation(
        payload={
            "prediction_rows": [
                {
                    "label": "meter",
                    "confidence": 0.91,
                    "xyxy": [600, 200, 900, 800],
                }
            ],
            "image_size": [1000, 1000],
            "target_labels": ["meter"],
            "visibility_score": 0.7,
            "readable_ratio": 0.8,
        },
        guidance_thresholds=thresholds,
        text_generator=FakeTextGenerator(),
    )

    assert result["guidance_action"] == "hold_still"


def test_capture_evidence_usecase_adds_item_and_assesses_bundle():
    stage_type = StageType("capture")
    bundle = EvidenceBundle(
        session_id=SessionId("session-001"),
        stage_type=stage_type,
        required_types=EvidenceRequirementPolicy().required_types_for(stage_type),
    )

    first_assessment = capture_evidence(
        bundle=bundle,
        item=EvidenceItem(
            item_id="evidence-001",
            evidence_type="screenshot",
            uri="captures/frame-001.jpg",
        ),
    )
    assert first_assessment.acceptable is False

    second_assessment = capture_evidence(
        bundle=bundle,
        item=EvidenceItem(
            item_id="evidence-002",
            evidence_type="clip",
            uri="captures/clip-001.mp4",
        ),
    )

    assert second_assessment.acceptable is True


def test_build_observation_payload_merges_detections_and_quality_signal():
    payload = build_observation_payload(
        prediction_rows=[
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]}
        ],
        image_size=(1000, 1000),
        target_labels=("meter",),
        visibility_score=0.85,
        readable_ratio=0.8,
    )

    assert payload == {
        "prediction_rows": [
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]}
        ],
        "image_size": [1000, 1000],
        "target_labels": ["meter"],
        "visibility_score": 0.85,
        "readable_ratio": 0.8,
    }


def test_build_raw_perception_payload_from_image_reads_size_and_exports_rows(
    tmp_path,
):
    image_path = tmp_path / "meter.png"
    Image.new("RGB", (1000, 1000), color="white").save(image_path)

    class FakeDetector:
        def predict(self, *, image_path, image_size, target_labels):
            assert str(image_path).endswith("meter.png")
            assert image_size == (1000, 1000)
            assert target_labels == ("meter",)
            return [
                DetectionCandidate(
                    label="meter",
                    confidence=0.91,
                    bbox=BoundingBox(x1=0.6, y1=0.2, x2=0.9, y2=0.8),
                )
            ]

    payload = build_raw_perception_payload_from_image(
        image_path=str(image_path),
        target_labels=("meter",),
        detector_adapter=FakeDetector(),
    )

    assert payload == {
        "detections": {
            "prediction_rows": [
                {
                    "label": "meter",
                    "confidence": 0.91,
                    "xyxy": [600.0, 200.0, 900.0, 800.0],
                }
            ],
            "image_size": [1000, 1000],
            "target_labels": ["meter"],
        },
        "vjepa": {
            "visibility_score": 1.0,
            "readable_ratio": 1.0,
        },
    }


def test_build_raw_perception_payload_from_clip_organizes_sampled_frames(
    tmp_path,
):
    clip_path = tmp_path / "inspection.mp4"
    clip_path.write_text("placeholder", encoding="utf-8")

    class FakeSampler:
        def sample(self, *, clip_path):
            assert clip_path.endswith("inspection.mp4")
            return {
                "frames": ["frame-0", "frame-5", "frame-9"],
                "frame_indices": [0, 5, 9],
                "image_size": (1280, 720),
            }

    class FakeDetector:
        def predict_sampled_frames(self, *, sampled_frames, image_size, target_labels):
            assert sampled_frames == ["frame-0", "frame-5", "frame-9"]
            assert image_size == (1280, 720)
            assert target_labels == ("meter",)
            return {
                "prediction_rows": [
                    {
                        "label": "meter",
                        "confidence": 0.95,
                        "xyxy": [320.0, 80.0, 980.0, 640.0],
                    }
                ]
            }

    class FakeClipScorer:
        def score_clip(self, *, sampled_frames, frame_indices, image_size):
            assert sampled_frames == ["frame-0", "frame-5", "frame-9"]
            assert frame_indices == [0, 5, 9]
            assert image_size == (1280, 720)
            return {
                "visibility_score": 0.86,
                "readable_ratio": 0.79,
                "stability_score": 0.91,
                "alignment_score": 0.84,
            }

    payload = build_raw_perception_payload_from_clip(
        clip_path=str(clip_path),
        target_labels=("meter",),
        sampler=FakeSampler(),
        detector=FakeDetector(),
        clip_scorer=FakeClipScorer(),
    )

    assert payload == {
        "detections": {
            "prediction_rows": [
                {
                    "label": "meter",
                    "confidence": 0.95,
                    "xyxy": [320.0, 80.0, 980.0, 640.0],
                }
            ],
            "image_size": [1280, 720],
            "target_labels": ["meter"],
        },
        "vjepa": {
            "visibility_score": 0.86,
            "readable_ratio": 0.79,
            "stability_score": 0.91,
            "alignment_score": 0.84,
        },
    }


def test_build_raw_perception_payload_from_clip_keeps_best_frame_and_clip_scores(
    tmp_path,
):
    clip_path = tmp_path / "inspection.mp4"
    clip_path.write_text("placeholder", encoding="utf-8")

    class FakeSampler:
        def sample(self, *, clip_path):
            assert clip_path.endswith("inspection.mp4")
            return SampledClip(
                frames=["frame-0", "frame-5", "frame-9"],
                frame_indices=[0, 5, 9],
                image_size=(1280, 720),
            )

    class FakeDetector:
        def predict_sampled_frames(self, *, sampled_frames, image_size, target_labels):
            assert sampled_frames == ["frame-0", "frame-5", "frame-9"]
            assert image_size == (1280, 720)
            assert target_labels == ("meter",)
            return {
                "prediction_rows": [
                    {
                        "label": "meter",
                        "confidence": 0.97,
                        "xyxy": [300.0, 90.0, 1000.0, 650.0],
                    }
                ],
                "best_frame_index": 1,
            }

    class FakeClipScorer:
        def score_clip(self, *, sampled_frames, frame_indices, image_size):
            assert sampled_frames == ["frame-0", "frame-5", "frame-9"]
            assert frame_indices == [0, 5, 9]
            assert image_size == (1280, 720)
            return {
                "visibility_score": 0.88,
                "readable_ratio": 0.81,
                "stability_score": 0.92,
                "alignment_score": 0.87,
            }

    payload = build_raw_perception_payload_from_clip(
        clip_path=str(clip_path),
        target_labels=("meter",),
        sampler=FakeSampler(),
        detector=FakeDetector(),
        clip_scorer=FakeClipScorer(),
    )

    assert payload == {
        "detections": {
            "prediction_rows": [
                {
                    "label": "meter",
                    "confidence": 0.97,
                    "xyxy": [300.0, 90.0, 1000.0, 650.0],
                }
            ],
            "image_size": [1280, 720],
            "target_labels": ["meter"],
            "best_frame_index": 1,
        },
        "vjepa": {
            "visibility_score": 0.88,
            "readable_ratio": 0.81,
            "stability_score": 0.92,
            "alignment_score": 0.87,
        },
    }


def test_build_raw_perception_payload_from_clip_accepts_dataclass_quality_signal(
    tmp_path,
):
    clip_path = tmp_path / "inspection.mp4"
    clip_path.write_text("placeholder", encoding="utf-8")

    class FakeSampler:
        def sample(self, *, clip_path):
            return SampledClip(
                frames=["frame-0", "frame-5"],
                frame_indices=[0, 5],
                image_size=(1280, 720),
            )

    class FakeDetector:
        def predict_sampled_frames(self, *, sampled_frames, image_size, target_labels):
            return {
                "prediction_rows": [
                    {
                        "label": "person",
                        "confidence": 0.93,
                        "xyxy": [280.0, 100.0, 1030.0, 680.0],
                    }
                ],
                "best_frame_index": 0,
            }

    class FakeClipScorer:
        def score_clip(self, *, sampled_frames, frame_indices, image_size):
            return ClipQualitySignal(
                visibility_score=0.9,
                readable_ratio=0.83,
                stability_score=0.94,
                alignment_score=0.89,
            )

    payload = build_raw_perception_payload_from_clip(
        clip_path=str(clip_path),
        target_labels=("person",),
        sampler=FakeSampler(),
        detector=FakeDetector(),
        clip_scorer=FakeClipScorer(),
    )

    assert payload["vjepa"] == {
        "visibility_score": 0.9,
        "readable_ratio": 0.83,
        "stability_score": 0.94,
        "alignment_score": 0.89,
    }


def test_run_runtime_observation_returns_structured_runtime_result():
    from allinone.application.runtime.run_runtime_observation import (
        run_runtime_observation,
    )

    class FakePromptBuilder:
        def build_guidance_explanation_prompt(self, *, observation, decision):
            assert observation.visibility_score == 0.85
            assert observation.readable_ratio == 0.8
            assert decision.action.value == "left"
            assert decision.reason == "target_shifted_right"
            return "prompt-for-qwen"

    class FakeTextGenerator:
        def generate(self, prompt: str):
            assert prompt == "prompt-for-qwen"
            return (
                """{
                    "operator_message": "请向左移动，让仪表回到画面中央。",
                    "suggested_action": "left",
                    "confidence": 0.82,
                    "evidence_focus": "确保整个表盘完整可见"
                }""",
                "fake-qwen",
            )

    result = run_runtime_observation(
        payload={
            "prediction_rows": [
                {
                    "label": "meter",
                    "confidence": 0.91,
                    "xyxy": [600, 200, 900, 800],
                }
            ],
            "image_size": [1000, 1000],
            "target_labels": ["meter"],
            "visibility_score": 0.85,
            "readable_ratio": 0.8,
        },
        prompt_builder=FakePromptBuilder(),
        text_generator=FakeTextGenerator(),
    )

    assert result == {
        "guidance_action": "left",
        "reason": "target_shifted_right",
        "language_action": "left",
        "confidence": 0.82,
        "operator_message": "请向左移动，让仪表回到画面中央。",
        "evidence_focus": "确保整个表盘完整可见",
        "language_source": "fake-qwen",
    }


def test_run_runtime_observation_handles_missing_target_without_crashing():
    from allinone.application.runtime.run_runtime_observation import (
        run_runtime_observation,
    )

    class FailIfCalledTextGenerator:
        def generate(self, prompt: str):
            raise AssertionError(f"text generator should not run: {prompt}")

    result = run_runtime_observation(
        payload={
            "prediction_rows": [],
            "image_size": [1000, 1000],
            "target_labels": ["meter"],
            "visibility_score": 0.22,
            "readable_ratio": 0.18,
        },
        text_generator=FailIfCalledTextGenerator(),
    )

    assert result == {
        "guidance_action": "hold_still",
        "reason": "target_not_detected",
        "language_action": "hold_still",
        "confidence": 0.0,
        "operator_message": "未检测到目标，请移动镜头搜索目标区域。",
        "evidence_focus": "先让目标进入画面，再继续判断取景质量",
        "language_source": "fallback",
    }


def test_qwen_runtime_text_generator_reuses_qwen_client(tmp_path, monkeypatch):
    from allinone.application.runtime.run_runtime_observation import (
        QwenRuntimeTextGenerator,
    )

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
        "allinone.application.runtime.run_runtime_observation.QwenGatewayConfig",
        FakeGatewayConfig,
        raising=False,
    )
    monkeypatch.setattr(
        "allinone.application.runtime.run_runtime_observation.QwenGateway",
        FakeGateway,
        raising=False,
    )

    generator = QwenRuntimeTextGenerator(recipe)

    assert generator.generate("prompt-1")[1] == "service"
    assert generator.generate("prompt-2")[1] == "service"
    assert calls == {
        "from_recipe": 1,
        "gateway_init": 1,
        "generate_text": 2,
    }
