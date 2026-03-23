import sys
from types import SimpleNamespace
from pathlib import Path

from allinone.domain.guidance.entities import GuidanceDecision
from allinone.domain.perception.entities import PerceptionObservation
from allinone.domain.shared.value_objects import BoundingBox, CenterOffset, PromptAction
from allinone.infrastructure.language.qwen.client import QwenClient
from allinone.infrastructure.language.qwen.prompt_builder import QwenPromptBuilder
from allinone.infrastructure.language.qwen.structured_output import (
    GuidanceExplanation,
    QwenStructuredOutputParser,
)
from tests._repo import repo_root


def _build_observation() -> PerceptionObservation:
    return PerceptionObservation(
        visibility_score=0.85,
        readable_ratio=0.8,
        fill_ratio=0.18,
        center_offset=CenterOffset(dx=0.25, dy=0.0),
        roi=BoundingBox(x1=0.6, y1=0.2, x2=0.9, y2=0.8),
    )


def _build_decision() -> GuidanceDecision:
    return GuidanceDecision(
        action=PromptAction("left"),
        reason="target_shifted_right",
    )


def test_prompt_builder_includes_guidance_context():
    prompt = QwenPromptBuilder().build_guidance_explanation_prompt(
        observation=_build_observation(),
        decision=_build_decision(),
    )

    assert "suggested_action" in prompt
    assert "target_shifted_right" in prompt
    assert "0.25" in prompt
    assert "left" in prompt


def test_structured_output_parser_extracts_guidance_explanation():
    raw_text = """```json
    {
      "operator_message": "请向左移动，让仪表回到画面中央。",
      "suggested_action": "left",
      "confidence": 0.82,
      "evidence_focus": "确保整个表盘完整可见"
    }
    ```"""

    parsed = QwenStructuredOutputParser().parse_guidance_explanation(raw_text)

    assert parsed == GuidanceExplanation(
        operator_message="请向左移动，让仪表回到画面中央。",
        suggested_action="left",
        confidence=0.82,
        evidence_focus="确保整个表盘完整可见",
    )


def test_qwen_client_loads_recipe_and_builds_generation_request():
    recipe = repo_root() / "configs/model_recipes/qwen35_9b.yaml"
    client = QwenClient.from_recipe(recipe)

    request = client.build_generation_request("hello")

    assert client.model_id == "Qwen/Qwen3.5-9B"
    assert client.model_path.endswith("/models/qwen/Qwen3.5-9B")
    assert request.prompt == "hello"
    assert request.max_new_tokens == 256
    assert request.temperature == 0.2


def test_qwen_client_builds_local_pipeline_with_auto_dtype(monkeypatch):
    captured: dict[str, object] = {}
    fake_tokenizer = object()
    fake_model = object()

    class FakePipeline:
        def __call__(self, prompt: str, **kwargs):
            captured["prompt"] = prompt
            captured["call_kwargs"] = kwargs
            return [{"generated_text": '{"operator_message":"ok"}'}]

    def fake_pipeline(task: str, **kwargs):
        captured["task"] = task
        captured["pipeline_kwargs"] = kwargs
        return FakePipeline()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            captured["tokenizer_model_path"] = model_path
            captured["tokenizer_kwargs"] = kwargs
            return fake_tokenizer

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            captured["model_model_path"] = model_path
            captured["model_kwargs"] = kwargs
            return fake_model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
            pipeline=fake_pipeline,
        ),
    )

    client = QwenClient(
        model_id="Qwen/Qwen3.5-9B",
        model_path="/tmp/qwen-runtime",
    )

    generated = client.generate_text("hello")

    assert generated == '{"operator_message":"ok"}'
    assert captured["task"] == "text-generation"
    assert captured["tokenizer_model_path"] == "/tmp/qwen-runtime"
    assert captured["tokenizer_kwargs"] == {
        "local_files_only": True,
    }
    assert captured["model_model_path"] == "/tmp/qwen-runtime"
    assert captured["model_kwargs"] == {
        "device_map": "auto",
        "local_files_only": True,
        "torch_dtype": "auto",
    }
    assert captured["pipeline_kwargs"] == {
        "model": fake_model,
        "tokenizer": fake_tokenizer,
    }
