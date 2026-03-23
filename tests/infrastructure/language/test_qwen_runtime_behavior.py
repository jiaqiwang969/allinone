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


def test_qwen_client_builds_local_runtime_with_auto_dtype(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, shape, moved_name: str) -> None:
            self.shape = shape
            self.moved_name = moved_name

        def to(self, device):
            captured.setdefault("tensor_moves", []).append((self.moved_name, device))
            return self

    fake_model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=None),
        generation_config=SimpleNamespace(pad_token_id=None),
    )
    fake_inputs = {
        "input_ids": FakeTensor((1, 2), "input_ids"),
        "attention_mask": FakeTensor((1, 2), "attention_mask"),
    }

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            captured["generation_config_kwargs"] = kwargs

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = None
            self.eos_token_id = 42
            self.pad_token = None
            self.eos_token = "</s>"

        def __call__(self, prompt: str, **kwargs):
            captured["prompt"] = prompt
            captured["tokenize_kwargs"] = kwargs
            return fake_inputs

        def decode(self, token_ids, **kwargs):
            captured["decode_token_ids"] = token_ids
            captured["decode_kwargs"] = kwargs
            return '{"operator_message":"ok"}'

    fake_tokenizer = FakeTokenizer()

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

    def fake_generate(**kwargs):
        captured["generate_kwargs"] = kwargs
        return [["prompt", "tokens", "generated", "text"]]

    fake_model.generate = fake_generate
    fake_model.device = "cuda:0"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
            GenerationConfig=FakeGenerationConfig,
        ),
    )

    client = QwenClient(
        model_id="Qwen/Qwen3.5-9B",
        model_path="/tmp/qwen-runtime",
    )

    generated = client.generate_text("hello")

    assert generated == '{"operator_message":"ok"}'
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
    assert captured["tokenize_kwargs"] == {
        "return_tensors": "pt",
    }
    assert captured["generation_config_kwargs"] == {
        "max_new_tokens": 256,
        "temperature": 0.2,
        "do_sample": True,
    }
    assert captured["generate_kwargs"] == {
        "input_ids": fake_inputs["input_ids"],
        "attention_mask": fake_inputs["attention_mask"],
        "generation_config": captured["generate_kwargs"]["generation_config"],
    }
    assert captured["tensor_moves"] == [
        ("input_ids", "cuda:0"),
        ("attention_mask", "cuda:0"),
    ]
    assert captured["decode_token_ids"] == ["generated", "text"]
    assert captured["decode_kwargs"] == {
        "skip_special_tokens": True,
    }
    assert fake_tokenizer.pad_token == "</s>"
    assert fake_model.config.pad_token_id == 42
    assert fake_model.generation_config.pad_token_id == 42


def test_qwen_client_generates_text_without_pipeline(monkeypatch):
    captured: dict[str, object] = {}
    fake_model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=7),
        generation_config=SimpleNamespace(pad_token_id=7),
        device="cpu",
    )
    fake_inputs = {
        "input_ids": SimpleNamespace(shape=(1, 1), to=lambda device: ["prompt"]),
    }

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            captured["generation_config_kwargs"] = kwargs

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = 7
            self.eos_token_id = 7
            self.pad_token = "<pad>"
            self.eos_token = "</s>"

        def __call__(self, prompt: str, **kwargs):
            captured["prompt"] = prompt
            return fake_inputs

        def decode(self, token_ids, **kwargs):
            return '{"operator_message":"ok"}'

    fake_tokenizer = FakeTokenizer()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            return fake_tokenizer

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            return fake_model

    fake_model.generate = lambda **kwargs: [["prompt", "generated"]]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
            GenerationConfig=FakeGenerationConfig,
        ),
    )

    client = QwenClient(
        model_id="Qwen/Qwen3.5-9B",
        model_path="/tmp/qwen-runtime",
    )

    generated = client.generate_text("hello")

    assert generated == '{"operator_message":"ok"}'
    assert captured["generation_config_kwargs"] == {
        "max_new_tokens": 256,
        "temperature": 0.2,
        "do_sample": True,
    }
    assert captured["prompt"] == "hello"


def test_qwen_client_from_recipe_returns_cached_instance(tmp_path):
    recipe = tmp_path / "qwen35_9b.yaml"
    recipe.write_text(
        "model_id: Qwen/Qwen3.5-9B\nruntime_path: /tmp/qwen-runtime\n",
        encoding="utf-8",
    )

    first = QwenClient.from_recipe(recipe)
    second = QwenClient.from_recipe(recipe)

    assert first is second


def test_qwen_client_cached_recipe_instance_reuses_pipeline(monkeypatch, tmp_path):
    captured = {
        "model_calls": 0,
        "tokenizer_calls": 0,
        "generate_calls": 0,
    }
    fake_model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=7),
        generation_config=SimpleNamespace(pad_token_id=7),
        device="cpu",
    )
    recipe = tmp_path / "qwen35_9b.yaml"
    recipe.write_text(
        "model_id: Qwen/Qwen3.5-9B\nruntime_path: /tmp/qwen-runtime\n",
        encoding="utf-8",
    )

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            pass

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = 7
            self.eos_token_id = 7
            self.pad_token = "<pad>"
            self.eos_token = "</s>"

        def __call__(self, prompt, **kwargs):
            return {
                "input_ids": SimpleNamespace(shape=(1, 1), to=lambda device: ["prompt"]),
            }

        def decode(self, token_ids, **kwargs):
            return '{"operator_message":"ok"}'

    fake_tokenizer = FakeTokenizer()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            captured["tokenizer_calls"] += 1
            return fake_tokenizer

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            captured["model_calls"] += 1
            return fake_model

    def fake_generate(**kwargs):
        captured["generate_calls"] += 1
        return [["prompt", "generated"]]

    fake_model.generate = fake_generate

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
            GenerationConfig=FakeGenerationConfig,
        ),
    )

    first = QwenClient.from_recipe(recipe)
    second = QwenClient.from_recipe(recipe)

    assert first.generate_text("hello") == '{"operator_message":"ok"}'
    assert second.generate_text("world") == '{"operator_message":"ok"}'
    assert captured == {
        "model_calls": 1,
        "tokenizer_calls": 1,
        "generate_calls": 2,
    }


def test_qwen_client_strips_think_block_before_structured_output(monkeypatch):
    fake_model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=7),
        generation_config=SimpleNamespace(pad_token_id=7),
        device="cpu",
    )

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            pass

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = 7
            self.eos_token_id = 7
            self.pad_token = "<pad>"
            self.eos_token = "</s>"

        def __call__(self, prompt: str, **kwargs):
            return {
                "input_ids": SimpleNamespace(shape=(1, 1), to=lambda device: ["prompt"]),
            }

        def decode(self, token_ids, **kwargs):
            return """<think>
先分析当前画面，再给出结论。
</think>
```json
{"operator_message":"ok"}
```"""

    fake_tokenizer = FakeTokenizer()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            return fake_tokenizer

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            return fake_model

    fake_model.generate = lambda **kwargs: [["prompt", "generated"]]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
            GenerationConfig=FakeGenerationConfig,
        ),
    )

    client = QwenClient(
        model_id="Qwen/Qwen3.5-9B",
        model_path="/tmp/qwen-runtime",
    )

    generated = client.generate_text("hello")

    assert generated == '```json\n{"operator_message":"ok"}\n```'


def test_qwen_client_strips_leading_thinking_process_text(monkeypatch):
    fake_model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=7),
        generation_config=SimpleNamespace(pad_token_id=7),
        device="cpu",
    )

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            pass

    class FakeTokenizer:
        def __init__(self) -> None:
            self.pad_token_id = 7
            self.eos_token_id = 7
            self.pad_token = "<pad>"
            self.eos_token = "</s>"

        def __call__(self, prompt: str, **kwargs):
            return {
                "input_ids": SimpleNamespace(shape=(1, 1), to=lambda device: ["prompt"]),
            }

        def decode(self, token_ids, **kwargs):
            return """Thinking Process:
1. 先判断目标位置。
2. 再输出 JSON。
```json
{"operator_message":"ok"}
```"""

    fake_tokenizer = FakeTokenizer()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            return fake_tokenizer

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            return fake_model

    fake_model.generate = lambda **kwargs: [["prompt", "generated"]]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModelForCausalLM,
            AutoTokenizer=FakeAutoTokenizer,
            GenerationConfig=FakeGenerationConfig,
        ),
    )

    client = QwenClient(
        model_id="Qwen/Qwen3.5-9B",
        model_path="/tmp/qwen-runtime",
    )

    generated = client.generate_text("hello")

    assert generated == '```json\n{"operator_message":"ok"}\n```'
