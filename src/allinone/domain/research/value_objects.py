"""Research value objects."""

from __future__ import annotations

from dataclasses import dataclass

from allinone.domain.shared.errors import DomainValidationError


@dataclass(frozen=True)
class ExperimentId:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise DomainValidationError("experiment id must not be empty")


@dataclass(frozen=True)
class MetricName:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise DomainValidationError("metric name must not be empty")
