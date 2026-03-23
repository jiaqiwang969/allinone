from allinone.application.runtime.ingest_observation_window import (
    ingest_observation_window,
)
from allinone.application.runtime.request_guidance_decision import (
    request_guidance_decision,
)
from allinone.domain.perception.entities import PerceptionObservation


def test_ingest_observation_window_returns_domain_observation():
    observation = ingest_observation_window(
        prediction_rows=[
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]},
            {"label": "meter", "confidence": 0.82, "xyxy": [200, 200, 500, 800]},
        ],
        image_size=(1000, 1000),
        target_labels=("meter",),
        visibility_score=0.85,
        readable_ratio=0.8,
    )

    assert isinstance(observation, PerceptionObservation)
    assert observation.center_offset.dx > 0


def test_ingest_observation_window_can_drive_guidance_decision():
    observation = ingest_observation_window(
        prediction_rows=[
            {"label": "meter", "confidence": 0.91, "xyxy": [600, 200, 900, 800]},
        ],
        image_size=(1000, 1000),
        target_labels=("meter",),
        visibility_score=0.85,
        readable_ratio=0.8,
    )

    decision = request_guidance_decision(observation)

    assert decision.action.value == "left"
