# Experiments

This directory stores experiment artifacts, manifests, and replay inputs for the
allinone research loop.

Sample runtime payloads live under `experiments/samples/` and can be passed to
`python3 -m allinone.interfaces.cli.main runtime-observation --input ...`.

Raw upstream perception examples also live under `experiments/samples/` and can
be converted with `python3 -m allinone.interfaces.cli.main build-observation-payload`.

Single-image raw perception payloads can also be generated from live YOLO
inference with:

`python3 -m allinone.interfaces.cli.main detect-image --image <image> --model <model> --targets <label1,label2> --output <raw.json>`

Clip-level raw perception payloads can be generated from `YOLO + V-JEPA` with:

`python3 -m allinone.interfaces.cli.main analyze-clip --clip <clip.mp4> --yolo-model <model> --vjepa-repo <repo> --vjepa-checkpoint <ckpt> --targets <label1,label2> --output <raw.json>`

Sensitive guidance replay datasets can be generated from a frozen raw payload with:

`python3 -m allinone.interfaces.cli.main build-guidance-replay-dataset --input-raw <raw.json> --output-dir experiments/generated/person_boundary_replay --target-label <label>`

The command writes:

- `raw/tight_center_boundary.json`
- `raw/direction_trigger_boundary.json`
- `raw/oversize_boundary.json`
- `manifest.jsonl`

Batch replay experiments can be launched with:

`python3 -m allinone.interfaces.cli.main run-experiment --manifest experiments/manifests/m400_phase1_demo.jsonl --run-dir experiments/runs/run-001 --candidate baseline --yolo-model <model> --vjepa-repo <repo> --vjepa-checkpoint <ckpt>`

The command writes:

- `manifest.jsonl`
- `results.jsonl`
- `summary.json`
- `raw/*.json`
- `payload/*.json`

Multiple candidate runs can then be judged with:

`python3 -m allinone.interfaces.cli.main judge-experiment --experiment-id exp-judge-001 --hypothesis "compare candidate runs" --target-metric guidance_success_rate --candidate-run baseline=experiments/runs/run-001 --candidate-run candidate-a=experiments/runs/run-002 --output experiments/judgements/exp-judge-001.json`

The judgement output contains:

- `experiment_id`
- `target_metric`
- `status`
- `candidate_scores`
- `best_candidate_name`

One-step runtime policy search can be launched with:

`python3 -m allinone.interfaces.cli.main run-research-step --experiment-id exp-loop-001 --hypothesis "tighten guidance thresholds" --target-metric guidance_success_rate --manifest experiments/manifests/m400_phase1_demo.jsonl --base-policy configs/runtime_policies/m400_default.json --candidate-count 3 --run-root experiments/research/exp-loop-001 --output experiments/research/exp-loop-001/summary.json --yolo-model <model> --vjepa-repo <repo> --vjepa-checkpoint <ckpt>`

The command writes:

- `candidate_policies/*.json`
- `runs/<candidate>/...`
- `judgement.json`
- `summary.json`
