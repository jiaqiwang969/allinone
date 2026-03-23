"""Load frozen raw perception payloads for offline replay."""

from __future__ import annotations

import json
from pathlib import Path


class JsonRawPayloadLoader:
    """Read raw perception payloads from JSON files."""

    def load(self, path: str) -> dict[str, object]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
