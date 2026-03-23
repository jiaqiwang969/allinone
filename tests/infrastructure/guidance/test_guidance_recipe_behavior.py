import json

from allinone.domain.guidance.services import GuidanceThresholds
from allinone.infrastructure.guidance.policy_recipe import RuntimePolicyRecipeStore


def test_runtime_policy_recipe_store_loads_guidance_thresholds(tmp_path):
    recipe_path = tmp_path / "policy.json"
    recipe_path.write_text(
        json.dumps(
            {
                "guidance_thresholds": {
                    "centered_offset_max": 0.11,
                    "directional_offset_min": 0.22,
                    "ready_fill_ratio_max": 0.9,
                }
            }
        ),
        encoding="utf-8",
    )

    thresholds = RuntimePolicyRecipeStore().load_guidance_thresholds(recipe_path)

    assert thresholds == GuidanceThresholds(
        centered_offset_max=0.11,
        directional_offset_min=0.22,
        ready_fill_ratio_max=0.9,
    )
