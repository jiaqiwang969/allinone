from allinone.infrastructure.language.qwen.schemas import (
    QwenGatewayConfig,
    QwenServiceGenerateRequest,
    QwenServiceGenerateResponse,
)
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
