import pytest

from allinone.infrastructure.language.qwen.gateway import QwenGateway
from allinone.infrastructure.language.qwen.schemas import (
    QwenGatewayConfig,
    QwenServiceGenerateRequest,
    QwenServiceGenerateResponse,
)
from allinone.infrastructure.language.qwen.service_client import QwenServiceClient
from tests._repo import repo_root


def test_qwen_service_generate_request_schema_serializes_to_payload():
    request = QwenServiceGenerateRequest(
        prompt="请只输出 JSON",
        max_new_tokens=128,
        temperature=0.1,
    )

    assert request.to_payload() == {
        "prompt": "请只输出 JSON",
        "max_new_tokens": 128,
        "temperature": 0.1,
    }


def test_qwen_service_generate_response_schema_round_trips_from_payload():
    payload = {
        "text": '{"operator_message":"ok"}',
        "model_id": "Qwen/Qwen3.5-9B",
        "mode": "service",
    }

    response = QwenServiceGenerateResponse.from_payload(payload)

    assert response == QwenServiceGenerateResponse(
        text='{"operator_message":"ok"}',
        model_id="Qwen/Qwen3.5-9B",
        mode="service",
    )
    assert response.to_payload() == payload


def test_qwen_gateway_config_schema_loads_gateway_recipe():
    recipe = repo_root() / "configs/model_recipes/qwen_gateway.yaml"

    config = QwenGatewayConfig.from_recipe(recipe)

    assert config.mode == "auto"
    assert config.service_url == "http://127.0.0.1:8001"
    assert config.service_timeout_seconds == 30
    assert config.model_id == "Qwen/Qwen3.5-9B"
    assert config.runtime_path.endswith("/models/qwen/Qwen3.5-9B")
    assert config.device == "auto"
    assert config.max_new_tokens == 256
    assert config.temperature == 0.2


def test_qwen_gateway_prefers_service_when_service_is_healthy():
    calls = {
        "service_generate": 0,
        "local_runtime_available": 0,
        "local_generate": 0,
    }

    class FakeServiceClient:
        def is_service_available(self) -> bool:
            return True

        def generate_text(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
        ) -> str:
            calls["service_generate"] += 1
            assert prompt == "prompt-1"
            assert max_new_tokens == 128
            assert temperature == 0.1
            return '{"operator_message":"service"}'

    class FakeLocalClient:
        def is_runtime_available(self) -> bool:
            calls["local_runtime_available"] += 1
            return True

        def generate_text(self, prompt: str, **kwargs) -> str:
            calls["local_generate"] += 1
            return '{"operator_message":"local"}'

    gateway = QwenGateway(
        config=QwenGatewayConfig.from_recipe(
            repo_root() / "configs/model_recipes/qwen_gateway.yaml"
        ),
        service_client=FakeServiceClient(),
        local_client=FakeLocalClient(),
    )

    text, source = gateway.generate_text(
        "prompt-1",
        max_new_tokens=128,
        temperature=0.1,
    )

    assert text == '{"operator_message":"service"}'
    assert source == "service"
    assert calls == {
        "service_generate": 1,
        "local_runtime_available": 0,
        "local_generate": 0,
    }


def test_qwen_gateway_falls_back_to_local_when_service_is_unavailable():
    calls = {
        "local_runtime_available": 0,
        "local_generate": 0,
    }

    class FakeServiceClient:
        def is_service_available(self) -> bool:
            return False

        def generate_text(self, prompt: str, **kwargs) -> str:
            raise AssertionError("service path should not be used")

    class FakeLocalClient:
        def is_runtime_available(self) -> bool:
            calls["local_runtime_available"] += 1
            return True

        def generate_text(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
        ) -> str:
            calls["local_generate"] += 1
            assert prompt == "prompt-2"
            assert max_new_tokens == 64
            assert temperature == 0.3
            return '{"operator_message":"local"}'

    gateway = QwenGateway(
        config=QwenGatewayConfig.from_recipe(
            repo_root() / "configs/model_recipes/qwen_gateway.yaml"
        ),
        service_client=FakeServiceClient(),
        local_client=FakeLocalClient(),
    )

    text, source = gateway.generate_text(
        "prompt-2",
        max_new_tokens=64,
        temperature=0.3,
    )

    assert text == '{"operator_message":"local"}'
    assert source == "local"
    assert calls == {
        "local_runtime_available": 1,
        "local_generate": 1,
    }


def test_qwen_gateway_raises_when_no_runtime_path_is_available():
    class FakeServiceClient:
        def is_service_available(self) -> bool:
            return False

    class FakeLocalClient:
        def is_runtime_available(self) -> bool:
            return False

    gateway = QwenGateway(
        config=QwenGatewayConfig.from_recipe(
            repo_root() / "configs/model_recipes/qwen_gateway.yaml"
        ),
        service_client=FakeServiceClient(),
        local_client=FakeLocalClient(),
    )

    with pytest.raises(RuntimeError, match="No Qwen runtime path available"):
        gateway.generate_text("prompt-3")


def test_qwen_gateway_service_client_reports_health_and_sanitizes_output():
    calls = []

    def fake_request_json(
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        calls.append((method, path, payload))
        if path == "/health":
            return {"status": "ready"}
        if path == "/generate":
            assert payload == {
                "prompt": "prompt-4",
                "max_new_tokens": 32,
                "temperature": 0.0,
            }
            return {
                "text": "<think>先分析</think>\n```json\n{\"operator_message\":\"ok\"}\n```",
                "model_id": "Qwen/Qwen3.5-9B",
                "mode": "service",
            }
        raise AssertionError(f"unexpected path: {path}")

    client = QwenServiceClient(
        service_url="http://127.0.0.1:8001",
        timeout_seconds=30,
        request_json=fake_request_json,
    )

    assert client.is_service_available() is True
    assert (
        client.generate_text(
            "prompt-4",
            max_new_tokens=32,
            temperature=0.0,
        )
        == '```json\n{"operator_message":"ok"}\n```'
    )
    assert calls == [
        ("GET", "/health", None),
        (
            "POST",
            "/generate",
            {
                "prompt": "prompt-4",
                "max_new_tokens": 32,
                "temperature": 0.0,
            },
        ),
    ]
