import json
import threading
from urllib import request

from allinone.interfaces.cli import main as cli_main
from allinone.interfaces.qwen_service import build_qwen_service_server


def test_build_parser_includes_serve_qwen_command():
    parser = cli_main._build_parser()

    args = parser.parse_args(["serve-qwen", "--host", "127.0.0.1", "--port", "8001"])

    assert args.command == "serve-qwen"
    assert args.host == "127.0.0.1"
    assert args.port == 8001


def test_qwen_service_server_exposes_health_and_generate_endpoints():
    class FakeTextGenerator:
        model_id = "Qwen/Qwen3.5-9B"

        def generate_text(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
        ) -> str:
            assert prompt == "请输出 JSON"
            assert max_new_tokens == 32
            assert temperature == 0.0
            return "<think>先分析</think>\n```json\n{\"operator_message\":\"ok\"}\n```"

    server = build_qwen_service_server(
        host="127.0.0.1",
        port=0,
        text_generator=FakeTextGenerator(),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        health_response = request.urlopen(f"http://{host}:{port}/health", timeout=5)
        health_payload = json.loads(health_response.read().decode("utf-8"))
        assert health_payload == {
            "status": "ready",
            "model_id": "Qwen/Qwen3.5-9B",
        }

        generate_request = request.Request(
            f"http://{host}:{port}/generate",
            data=json.dumps(
                {
                    "prompt": "请输出 JSON",
                    "max_new_tokens": 32,
                    "temperature": 0.0,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        generate_response = request.urlopen(generate_request, timeout=5)
        generate_payload = json.loads(generate_response.read().decode("utf-8"))
        assert generate_payload == {
            "text": '```json\n{"operator_message":"ok"}\n```',
            "model_id": "Qwen/Qwen3.5-9B",
            "mode": "service",
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
