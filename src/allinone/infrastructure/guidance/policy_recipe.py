"""Runtime policy recipe persistence for guidance thresholds."""

from __future__ import annotations

import json
from pathlib import Path

from allinone.domain.guidance.services import GuidanceThresholds


class RuntimePolicyRecipeStore:
    """Load and store runtime policy recipes as JSON files."""

    def load_guidance_thresholds(
        self,
        recipe_path: str | Path,
    ) -> GuidanceThresholds:
        payload = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
        thresholds = payload.get("guidance_thresholds", {})
        return GuidanceThresholds(
            centered_offset_max=float(
                thresholds.get(
                    "centered_offset_max",
                    GuidanceThresholds.centered_offset_max,
                )
            ),
            directional_offset_min=float(
                thresholds.get(
                    "directional_offset_min",
                    GuidanceThresholds.directional_offset_min,
                )
            ),
            ready_fill_ratio_max=float(
                thresholds.get(
                    "ready_fill_ratio_max",
                    GuidanceThresholds.ready_fill_ratio_max,
                )
            ),
        )

    def write_guidance_thresholds(
        self,
        *,
        recipe_path: str | Path,
        thresholds: GuidanceThresholds,
    ) -> Path:
        destination = Path(recipe_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "guidance_thresholds": {
                        "centered_offset_max": thresholds.centered_offset_max,
                        "directional_offset_min": thresholds.directional_offset_min,
                        "ready_fill_ratio_max": thresholds.ready_fill_ratio_max,
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return destination
