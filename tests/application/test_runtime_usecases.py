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
from allinone.application.runtime.request_guidance_decision import (
    request_guidance_decision,
)
from allinone.application.session.open_session import open_session
from allinone.domain.evidence.entities import EvidenceBundle, EvidenceItem
from allinone.domain.evidence.policies import EvidenceRequirementPolicy
from allinone.domain.guidance.entities import GuidanceDecision
from allinone.domain.perception.entities import PerceptionObservation
from allinone.domain.session.entities import WorkSession
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
