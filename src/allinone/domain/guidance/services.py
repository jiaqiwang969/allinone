"""Guidance services."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.guidance.entities import GuidanceDecision
from allinone.domain.perception.entities import PerceptionObservation
from allinone.domain.shared.value_objects import PromptAction


@dataclass(frozen=True)
class GuidanceThresholds:
    centered_offset_max: float = 0.09
    directional_offset_min: float = 0.18
    ready_fill_ratio_max: float = 0.85


class GuidancePolicyService:
    def __init__(self, thresholds: GuidanceThresholds | None = None) -> None:
        self.thresholds = thresholds or GuidanceThresholds()

    def decide(self, observation: PerceptionObservation) -> GuidanceDecision:
        if observation.fill_ratio >= self.thresholds.ready_fill_ratio_max:
            return GuidanceDecision(
                action=PromptAction("backward"),
                reason="target_too_large",
            )
        if observation.center_offset.dx >= self.thresholds.directional_offset_min:
            return GuidanceDecision(
                action=PromptAction("left"),
                reason="target_shifted_right",
            )
        if observation.center_offset.dx <= -self.thresholds.directional_offset_min:
            return GuidanceDecision(
                action=PromptAction("right"),
                reason="target_shifted_left",
            )
        if abs(observation.center_offset.dx) <= self.thresholds.centered_offset_max and abs(
            observation.center_offset.dy
        ) <= self.thresholds.centered_offset_max:
            return GuidanceDecision(
                action=PromptAction("hold_still"),
                reason="fully_centered",
            )
        return GuidanceDecision(
            action=PromptAction("hold_still"),
            reason="stabilize_before_capture",
        )
