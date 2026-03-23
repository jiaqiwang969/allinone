"""HTTP service host for long-lived Qwen inference."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from allinone.infrastructure.language.qwen.client import QwenClient


class _QwenServiceHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        text_generator: object,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.text_generator = text_generator


class _QwenServiceHandler(BaseHTTPRequestHandler):
    server: _QwenServiceHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._write_json({"error": "not_found"}, status_code=404)
            return
        self._write_json(
            {
                "status": "ready",
                "model_id": str(getattr(self.server.text_generator, "model_id", "unknown")),
            }
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/generate":
            self._write_json({"error": "not_found"}, status_code=404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        payload = json.loads(raw_body or "{}")
        text = self.server.text_generator.generate_text(
            str(payload["prompt"]),
            max_new_tokens=int(payload["max_new_tokens"]),
            temperature=float(payload["temperature"]),
        )
        self._write_json(
            {
                "text": QwenClient.sanitize_generated_text(str(text)),
                "model_id": str(getattr(self.server.text_generator, "model_id", "unknown")),
                "mode": "service",
            }
        )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, payload: dict[str, object], *, status_code: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def build_qwen_service_server(
    *,
    host: str,
    port: int,
    text_generator: object,
) -> ThreadingHTTPServer:
    return _QwenServiceHTTPServer(
        (host, port),
        _QwenServiceHandler,
        text_generator=text_generator,
    )
