"""Gateway for selecting Qwen service or local runtime."""

from __future__ import annotations

from allinone.infrastructure.language.qwen.client import QwenClient
from allinone.infrastructure.language.qwen.schemas import QwenGatewayConfig
from allinone.infrastructure.language.qwen.service_client import QwenServiceClient


class QwenGateway:
    """Unified entrypoint for service-backed or local Qwen generation."""

    def __init__(
        self,
        *,
        config: QwenGatewayConfig,
        service_client: object | None = None,
        local_client: object | None = None,
    ) -> None:
        self.config = config
        self._service_client = service_client
        self._local_client = local_client

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, str]:
        resolved_max_new_tokens = (
            self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        )
        resolved_temperature = (
            self.config.temperature if temperature is None else temperature
        )
        if self.config.mode == "service":
            service_client = self._resolve_service_client()
            if not service_client.is_service_available():
                raise RuntimeError("Qwen service is not available")
            return (
                service_client.generate_text(
                    prompt,
                    max_new_tokens=resolved_max_new_tokens,
                    temperature=resolved_temperature,
                ),
                "service",
            )
        if self.config.mode == "local":
            local_client = self._resolve_local_client()
            if not local_client.is_runtime_available():
                raise RuntimeError("No Qwen runtime path available")
            return (
                local_client.generate_text(
                    prompt,
                    max_new_tokens=resolved_max_new_tokens,
                    temperature=resolved_temperature,
                ),
                "local",
            )

        service_client = self._resolve_service_client()
        if service_client.is_service_available():
            return (
                service_client.generate_text(
                    prompt,
                    max_new_tokens=resolved_max_new_tokens,
                    temperature=resolved_temperature,
                ),
                "service",
            )

        local_client = self._resolve_local_client()
        if local_client.is_runtime_available():
            return (
                local_client.generate_text(
                    prompt,
                    max_new_tokens=resolved_max_new_tokens,
                    temperature=resolved_temperature,
                ),
                "local",
            )
        raise RuntimeError("No Qwen runtime path available")

    def _resolve_service_client(self) -> object:
        if self._service_client is None:
            self._service_client = QwenServiceClient(
                service_url=self.config.service_url,
                timeout_seconds=self.config.service_timeout_seconds,
            )
        return self._service_client

    def _resolve_local_client(self) -> object:
        if self._local_client is None:
            self._local_client = QwenClient(
                model_id=self.config.model_id,
                model_path=self.config.runtime_path,
                device=self.config.device,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
        return self._local_client
