"""Guidance entities."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.shared.value_objects import PromptAction


@dataclass(frozen=True)
class GuidanceDecision:
    action: PromptAction
    reason: str
