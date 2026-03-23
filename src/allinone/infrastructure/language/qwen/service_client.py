"""Service client for remote Qwen generation."""

from __future__ import annotations

import json
from typing import Callable
from urllib import request

from allinone.infrastructure.language.qwen.client import QwenClient
from allinone.infrastructure.language.qwen.schemas import (
    QwenServiceGenerateRequest,
    QwenServiceGenerateResponse,
)


class QwenServiceClient:
    """Call a long-lived Qwen service over HTTP."""

    def __init__(
        self,
        *,
        service_url: str,
        timeout_seconds: int,
        request_json: Callable[[str, str, dict[str, object] | None], dict[str, object]]
        | None = None,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._request_json = request_json or self._default_request_json

    def is_service_available(self) -> bool:
        try:
            payload = self._request_json("GET", "/health", None)
        except RuntimeError:
            return False
        return str(payload.get("status", "")) == "ready"

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        payload = self._request_json(
            "POST",
            "/generate",
            QwenServiceGenerateRequest(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            ).to_payload(),
        )
        response = QwenServiceGenerateResponse.from_payload(payload)
        return QwenClient.sanitize_generated_text(response.text)

    def _default_request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        http_request = request.Request(
            f"{self.service_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - exercised via caller behavior
            raise RuntimeError("unable to reach qwen service") from exc
