"""Request a guidance decision from the runtime loop."""

from __future__ import annotations

from allinone.domain.guidance.entities import GuidanceDecision
from allinone.domain.guidance.services import GuidancePolicyService, GuidanceThresholds
from allinone.domain.perception.entities import PerceptionObservation


def request_guidance_decision(
    observation: PerceptionObservation,
    *,
    policy_service: GuidancePolicyService | None = None,
    guidance_thresholds: GuidanceThresholds | None = None,
) -> GuidanceDecision:
    """Translate the current perception observation into a prompt action."""
    service = policy_service or GuidancePolicyService(thresholds=guidance_thresholds)
    return service.decide(observation)
