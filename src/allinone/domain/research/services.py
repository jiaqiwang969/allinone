"""Research services."""

from __future__ import annotations

from allinone.domain.research.entities import CandidateEvaluation, ExperimentRun
from allinone.domain.research.errors import ExperimentStateError


class ExperimentSelectionService:
    def select_best(self, run: ExperimentRun) -> CandidateEvaluation:
        best = run.best_evaluation
        if best is None:
            raise ExperimentStateError("cannot select best candidate without evaluations")
        return best
