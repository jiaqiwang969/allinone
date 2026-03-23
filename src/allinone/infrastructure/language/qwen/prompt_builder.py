"""Prompt builder boundary for Qwen language tasks."""

from __future__ import annotations

from allinone.domain.guidance.entities import GuidanceDecision
from allinone.domain.perception.entities import PerceptionObservation


class QwenPromptBuilder:
    """Build Qwen prompts around guidance and evidence tasks."""

    def build_guidance_explanation_prompt(
        self,
        *,
        observation: PerceptionObservation,
        decision: GuidanceDecision,
    ) -> str:
        return (
            "你是工业远程质检助手。请基于当前视频取景状态，输出 JSON，字段必须包含："
            "operator_message, suggested_action, confidence, evidence_focus。\n"
            f"visibility_score={observation.visibility_score:.2f}\n"
            f"readable_ratio={observation.readable_ratio:.2f}\n"
            f"fill_ratio={observation.fill_ratio:.2f}\n"
            f"center_offset_dx={observation.center_offset.dx:.2f}\n"
            f"center_offset_dy={observation.center_offset.dy:.2f}\n"
            f"suggested_action={decision.action.value}\n"
            f"decision_reason={decision.reason}\n"
            "请使用简洁中文，直接告诉工人下一步该怎么移动镜头，并指出证据采集重点。"
        )
