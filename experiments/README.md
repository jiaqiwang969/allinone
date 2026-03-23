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
