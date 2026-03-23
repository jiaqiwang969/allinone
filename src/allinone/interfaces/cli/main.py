"""CLI entrypoint for allinone."""

from __future__ import annotations

import argparse

from allinone.application.research.register_experiment import register_experiment
from allinone.application.runtime.request_guidance_decision import (
    request_guidance_decision,
)
from allinone.domain.perception.entities import PerceptionObservation
from allinone.domain.shared.value_objects import BoundingBox, CenterOffset
from allinone.infrastructure.research.autoresearch.replay_adapter import (
    AutoresearchReplayAdapter,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="allinone")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    subparsers.add_parser("guidance-smoke")
    subparsers.add_parser("research-smoke")
    return parser


def _run_guidance_smoke() -> int:
    observation = PerceptionObservation(
        visibility_score=0.7,
        readable_ratio=0.8,
        fill_ratio=0.5,
        center_offset=CenterOffset(dx=0.25, dy=0.0),
        roi=BoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9),
    )
    decision = request_guidance_decision(observation)
    print(f"guidance_action={decision.action.value} reason={decision.reason}")
    return 0


def _run_research_smoke() -> int:
    run = register_experiment(
        experiment_id="exp-smoke-001",
        hypothesis="compare baseline and candidate guidance policies",
        target_metric="guidance_success_rate",
        candidate_names=["baseline", "candidate-a"],
    )
    payload = AutoresearchReplayAdapter().build_payload(run)
    candidate_names = ",".join(payload["candidate_names"])
    print(
        f"experiment_id={payload['experiment_id']} "
        f"target_metric={payload['target_metric']} "
        f"candidate_names={candidate_names}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the allinone CLI."""
    args = _build_parser().parse_args(argv)
    if args.command == "guidance-smoke":
        return _run_guidance_smoke()
    if args.command == "research-smoke":
        return _run_research_smoke()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
