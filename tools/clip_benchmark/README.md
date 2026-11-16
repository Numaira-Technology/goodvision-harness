# Native CLIP / SigLIP2 controlled rerun

Selectively imported from draft PR #62, commit
`99d515a3f561460561d21e50cff994ad85f0fd99`; no merge or PR creation.
Variant: `native-clip-1fps-midpoint-v1`. Original sources are retained in the
local integration evidence directory. This is not the language/Luna v2 protocol.

Preserved: pinned model revisions and six text pairs, official image processor,
FP32 normalized cosine difference, strict `margin > threshold`, one frame per
inference, clip maximum, original paced feeder and primary/load-only distinction,
all-question accuracy denominator, unknowns and partial-test behavior.
SigLIP2 still lowercases its text pairs. Labels are never sent to the encoder.

Changes: interval 0.8 -> 1.0 seconds (1.25 -> 1 FPS), staged execution, lossless
score artifacts with code/config/media/split hashes, source-group sidecar join,
strict validation completeness, finite optimal interval midpoint selection,
new/old/zero-threshold comparisons from one set of test scores. The midpoint
rule comes from the archived November 1 analysis, not a new meeting decision.
Bag thresholds remain zero/uncalibrated; bag reports are separate from base4.
The twelve paced local file slots include load-only repeated completed videos;
they do not establish twelve-camera end-to-end performance. Busy/late drops and
native input-to-result timing remain visible in completion.json. Incomplete
validation stops selection; a test with partial evidence retains native scoring
and must be reported alongside coverage and execution status.

## Configuration and offline preflight

Supply an explicit local JSON file with `interval_seconds: 1.0`, `slots: 12`,
`split_manifest`, `reference_thresholds` (model-keyed historical thresholds), and
`validation` / `test`, each containing `samples`, `manifest`, `media` (directory
list). Relative paths resolve from the configuration file. Absolute local paths
can be relocated for the GPU host: frozen content hashes, not directory names,
define compatibility. Actual configuration bytes are also recorded.

```sh
python -I tools/clip_benchmark/run.py preflight --config /path/config.json \
  --decode --output /new/path/preflight.json
```

This hashes all frozen media/questions/labels, joins both split mappings, checks
native frame counts, and optionally decodes every original frame without pacing.
It does not import Torch, use credentials, download models or send HTTP.
Unknown training/development membership and source-isolation limits remain in
the artifact. A pass establishes input readiness, not GPU readiness.

## After rental and inference authorization

`prepare.sh /new/task-root` creates an isolated environment without inherited
site packages. Torch 2.8.0 / torchvision 0.23.0 pins reuse the repository's neural
extra; CUDA 12.8 wheels and PR #62's processor dependencies still require an
on-host install/compatibility check. Save the resolved package inventory and
image/driver identity. This proposed environment is not an assertion of historical
environment identity. Do not run this preparation or download weights before
the user's rental/download authorization.

Run each model separately, releasing GPU memory between models. Preserve the
same environment between validation and test. The actual GPU must be a 4090
with 24 GB before the runner attempts model downloads.

```sh
python -I tools/clip_benchmark/run.py validation --config /path/config.json \
  --model openai/clip-vit-large-patch14 --cache-dir /path/cache --output /new/val
python -I tools/clip_benchmark/run.py select --config /path/config.json \
  --model openai/clip-vit-large-patch14 --validation-bundle /new/val/bundle.json \
  --output /new/selected
python -I tools/clip_benchmark/run.py test --config /path/config.json \
  --model openai/clip-vit-large-patch14 --cache-dir /path/cache \
  --thresholds /new/selected/frozen-thresholds.json \
  --thresholds-sha256 FROZEN_HASH --output /new/test
```

Repeat with `google/siglip2-base-patch16-224`. Outputs refuse overwrites.
Threshold selection needs all scheduled primary observations and at least six
complete binary questions per base event, with both classes. It does not open
test predictions. The old thresholds are comparison-only, explicitly from the
historical 1.25 FPS experiment. The frozen threshold artifact must match current
model, code, input, split, prompt, sampling, precision and actual GPU/software
environment before test. Record real GPU memory peaks and missing hardware facts.

API and CLIP retain different original frame origins, image processing, text
formats and error denominators. Matching 1 FPS does not remove those differences.
Do not publish a model-only or leakage-free claim. Results remain local until
the user reviews a proposed shared-sheet diff; no automatic PR or publishing.
