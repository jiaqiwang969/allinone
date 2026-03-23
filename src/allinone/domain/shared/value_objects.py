"""Shared value objects."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.shared.errors import DomainValidationError


@dataclass(frozen=True)
class SessionId:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise DomainValidationError("session id must not be empty")


@dataclass(frozen=True)
class StageType:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise DomainValidationError("stage type must not be empty")


@dataclass(frozen=True)
class PromptAction:
    value: str

    ALLOWED = {
        "left",
        "right",
        "up",
        "down",
        "forward",
        "backward",
        "hold_still",
        "start_recording",
        "stop_recording",
    }

    def __post_init__(self) -> None:
        if self.value not in self.ALLOWED:
            raise DomainValidationError(f"unsupported prompt action: {self.value}")


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise DomainValidationError("bounding box coordinates must be increasing")


@dataclass(frozen=True)
class CenterOffset:
    dx: float
    dy: float
