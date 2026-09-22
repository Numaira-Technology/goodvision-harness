# Contributing

Run commands from the repository root. Python 3.10 or newer is required; CI
tests Python 3.11 on Linux and 3.12 on Windows. Node.js runs the dashboard
regressions, and FFmpeg runs the video encoder checks.

## Isolated development

Use a separate worktree when the existing checkout has ongoing work:

```bash
git fetch origin
git worktree add -b my-change .worktrees/my-change <base-branch>
cd .worktrees/my-change
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,webrtc,vlm]'
```

Replace `<base-branch>` with the branch you intend to change. On PowerShell,
create the environment with `python -m venv .venv` and activate it with
`.\.venv\Scripts\Activate.ps1`. Each worktree needs its own environment and
local settings. Copy `.env.example` to `.env` only when running the service;
the offline tests do not need service credentials or model downloads.

## Local development

After activating the environment above, start a console without model calls:

```bash
HARNESS_AUTOSTART=false HARNESS_DRY_RUN=true goodvision-harness-demo --host 127.0.0.1 --port 7900
```

In PowerShell, set `$env:HARNESS_AUTOSTART = 'false'` and
`$env:HARNESS_DRY_RUN = 'true'` before running the same launcher. Open
`http://127.0.0.1:7900`, choose Camera, enable it, then press Start. For a network
source, choose Network stream and enter a URL or use `HARNESS_STREAM_URL`.

`goodvision-harness-demo` and `goodvision-harness` serve the same console, with
different default ports; the latter defaults to 8080. `goodvision-harness-cli`
retains the window lifecycle, `vision_result` and final-state JSONL output.
For source containers, use the [deployment guide](docs/guides/deployment.md).

### Known-person gallery

For real local face recognition, install `python -m pip install -e '.[face]'`
and provision authorized detection/recognition weights under
`FACE_MODEL_ROOT/models/<configured-profile>/` (or use the configured first-use
download). Configure the [inference route](docs/guides/configuration.md) and
set `HARNESS_DRY_RUN=false` when ready for real inference.

Enroll a stable person ID, name and clear single-face image in the console's
known-person gallery; it persists at `FACE_GALLERY_PATH`. The face-only API mode
remains supported. See [event evidence](docs/architecture/event-evidence.md#people-and-continuity)
for identity continuity, overlays and source-time association, and
[reference memory](docs/guides/configuration.md#face-and-reference-memory)
for the separate optional remote-reference path.

Use [WINDOWS_HYBRID.md](WINDOWS_HYBRID.md) for DirectML and the
[runtime guide](docs/guides/runtime.md#optional-gpu-face-recognition) for CUDA.
Face models are not bundled; their licenses must cover the intended use.

## Checks

```bash
python tools/check_repo_hygiene.py
python -m pytest
git diff --check
```

The Python suite also invokes all dashboard checks when Node is installed;
there is no need to run them a second time. For a dashboard-only change, use
`node --test tests/*.cjs`. The [test guide](tests/README.md) lists focused runs
and the coverage expected when adding or consolidating tests.
CI installs Node, FFmpeg and the development extras explicitly. Real model, paid inference, remote GPU and
customer image checks are separate; use the relevant entries in the
[tools index](tools/README.md) and record which environment was exercised.

## What belongs in Git

Commit source, tests, schemas, deployment recipes, public configuration
templates, documentation and reviewed test fixtures. Keep secrets, local
environment files, credentials, private media, person galleries, model weights,
logs, caches, exported images and evaluation output outside Git.

- Use `.env` or a local `.env.*` file for credentials; only reviewed
  `.env*.example` templates belong in Git.
- Keep runtime state in `.goodvision/`, weights in `.models/`, private reference
  images in `memory/`, and generated runs in `output/` or `outputs/`. Use `dist/`
  for delivery staging and exported bundles.
- Put intentional public documentation images in `docs/assets/` and reviewed
  test media in `tests/fixtures/`. Include source/license information where
  relevant. Review image contents before adding them.
- Keep reviewed validation summaries and provenance in `docs/validation/`;
  raw captures, predictions and logs stay in an ignored output directory.
- Update `.dockerignore` alongside `.gitignore` when introducing a new local
  artifact directory, including the Paddle worker's separate build context.

Before committing, inspect `git status --short` and the staged diff. Use
`git check-ignore -v <path>` to explain an ignore rule. The hygiene command
checks the Git index against the repository's `.gitignore` files and fails if
an ignored file was tracked (including with `git add --force`). It does not
inspect file contents or Git history for secrets. Review new public templates
and fixtures even when the check passes. Personal global Git exclusions do not
affect the CI check.

## Organization and stable paths

Start with the [documentation index](docs/README.md) for the repository map.
Keep new guides under `docs/guides/`, API references under `docs/reference/`,
designs under `docs/architecture/`, and maintained validation conclusions under
`docs/validation/`. Update the existing topic instead of adding a report for
every run. Remove superseded notes and unused attachments; keep only necessary
reviewed evidence under `docs/validation/evidence/` and update affected links.

Root Compose files, Dockerfiles, environment examples and customer delivery
documents are stable entrypoints used by scripts, tests and operators. Review
their consumers before moving them. `config.yaml` and
`src/goodvision_harness/config.yaml` must remain identical: one is the checkout
default, the other is installed package data. An existing regression test
enforces this. Preserve the bundled font and its license/source files.
