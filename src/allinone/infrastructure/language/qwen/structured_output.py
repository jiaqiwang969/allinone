"""Structured output boundary for Qwen responses."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class GuidanceExplanation:
    operator_message: str
    suggested_action: str
    confidence: float
    evidence_focus: str


class QwenStructuredOutputParser:
    """Parse structured Qwen outputs into typed objects."""

    def parse_guidance_explanation(self, raw_text: str) -> GuidanceExplanation:
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        payload = json.loads(text)
        return GuidanceExplanation(
            operator_message=str(payload["operator_message"]),
            suggested_action=str(payload["suggested_action"]),
            confidence=float(payload["confidence"]),
            evidence_focus=str(payload["evidence_focus"]),
        )
