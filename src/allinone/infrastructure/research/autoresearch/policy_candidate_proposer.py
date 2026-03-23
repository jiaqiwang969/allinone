"""Rule-based runtime policy candidate generation."""

from __future__ import annotations


class RuleBasedPolicyCandidateProposer:
    """Generate deterministic policy mutations from a baseline recipe."""

    def propose_candidates(
        self,
        *,
        base_thresholds: dict[str, float],
        candidate_count: int,
    ) -> list[dict[str, object]]:
        candidates = [
            {
                "candidate_name": "baseline",
                "mutation": "baseline",
                "guidance_thresholds": dict(base_thresholds),
            }
        ]
        mutations = [
            ("tighten_center_window", self._tighten_center_window),
            ("earlier_direction_trigger", self._earlier_direction_trigger),
            (
                "allow_larger_target_before_backward",
                self._allow_larger_target_before_backward,
            ),
        ]
        for index, (mutation_name, mutation) in enumerate(
            mutations[: max(0, candidate_count - 1)],
            start=1,
        ):
            candidates.append(
                {
                    "candidate_name": f"candidate-{index}",
                    "mutation": mutation_name,
                    "guidance_thresholds": mutation(dict(base_thresholds)),
                }
            )
        return candidates

    def _tighten_center_window(
        self,
        thresholds: dict[str, float],
    ) -> dict[str, float]:
        thresholds["centered_offset_max"] = round(
            thresholds["centered_offset_max"] * 0.8,
            4,
        )
        return thresholds

    def _earlier_direction_trigger(
        self,
        thresholds: dict[str, float],
    ) -> dict[str, float]:
        thresholds["directional_offset_min"] = round(
            thresholds["directional_offset_min"] * 0.85,
            4,
        )
        return thresholds

    def _allow_larger_target_before_backward(
        self,
        thresholds: dict[str, float],
    ) -> dict[str, float]:
        thresholds["ready_fill_ratio_max"] = round(
            thresholds["ready_fill_ratio_max"] * 1.05,
            4,
        )
        return thresholds
