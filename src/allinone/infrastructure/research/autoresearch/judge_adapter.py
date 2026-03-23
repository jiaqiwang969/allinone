"""Adapter boundary for candidate judge integration."""

from __future__ import annotations

from allinone.domain.research.entities import CandidateEvaluation


class AutoresearchJudgeAdapter:
    """Convert autoresearch judge output into research domain objects."""

    def to_candidate_evaluation(
        self,
        *,
        candidate_name: str,
        score: float,
        summary: str,
    ) -> CandidateEvaluation:
        return CandidateEvaluation(
            candidate_name=candidate_name,
            score=score,
            summary=summary,
        )
