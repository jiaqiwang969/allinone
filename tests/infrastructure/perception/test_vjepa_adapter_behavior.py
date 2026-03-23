from allinone.infrastructure.perception.vjepa.encoder import (
    FrameQualitySignal,
    VJEPAEncoderAdapter,
)


def test_vjepa_adapter_normalizes_quality_signal():
    signal = VJEPAEncoderAdapter().normalize_quality_signal(
        {
            "visibility_score": 0.85,
            "readable_ratio": 0.8,
        }
    )

    assert signal == FrameQualitySignal(
        visibility_score=0.85,
        readable_ratio=0.8,
    )
