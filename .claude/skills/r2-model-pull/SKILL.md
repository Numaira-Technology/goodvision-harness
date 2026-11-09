---
name: r2-model-pull
description: >-
  Pull pinned local-weight vision models (VideoJev-4B, CLIP) from the Cloudflare R2 fixed
  store, locally or onto a rented GPU. Use for 从 R2 拉模型 / 下载 videojev / clip 权重 /
  模型从 R2 调取 / prefetch model weights / pull pinned weights / model_source / cloud_prefetch /
  R2 fallback to HF. Covers env setup, the pull command, loading with local_files_only, and the
  remote-over-SSH path. Only models uploaded to bucket goodvision-models;
  not API models (qwen/gemini/luna); credentials never in the repo. Companion: vast-gpu-rent
  (renting the GPU), docs/guides/model-storage-r2.md.
---

# r2-model-pull

Fetch revision-pinned vision weights from the R2 `goodvision-models` bucket instead of
HuggingFace. R2 downloads are staged, and every file is sha256-verified
against the bucket `MANIFEST.json`. HF is the automatic fallback when R2 creds are absent
or a file is missing.

The logic lives in `tools/model_source.py` (local pull) and `tools/cloud_prefetch.py`
(remote pull over SSH). This skill is how to drive them; it does not reimplement them.

## When
- You need VideoJev-4B or CLIP weights on this machine or on a rented GPU.
- A benchmark/demo runner should load from a local HF cache with `local_files_only=True`.
- You are standing up a rented GPU and want weights without a slow/flaky HF pull.

## How
Credentials: set `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`
in the gitignored `.env` (read-only token). `HF_TOKEN` is optional, for the HF fallback.
**`model_source` reads these from the environment and does NOT load `.env` itself** — export
them first, or it silently falls back to HF.

Local pull (order: verified-local → R2 → HF):
```bash
set -a; . ./.env; set +a    # load R2_* into the environment (required, else HF fallback)
PYTHONPATH=src python -m tools.model_source --model videojev --cache-dir <hf-cache>
PYTHONPATH=src python -m tools.model_source --model clip     --cache-dir <hf-cache>
# or an explicit repo:  --repo <org/name> --revision <40-char sha>
```
A successful R2 pull logs `ready from R2 <repo>@<rev>`; `downloading from HF <repo>` means it
fell back (creds missing/unreachable).

Load the pulled cache (no network):
```python
from transformers import AutoModel  # CLIP: CLIPModel / from neural_service
AutoModel.from_pretrained(MODEL, revision=REVISION, cache_dir="<hf-cache>",
                          local_files_only=True, trust_remote_code=True)  # trust_remote_code: VideoJev only
```

Remote (onto a rented GPU over SSH): `tools/cloud_prefetch.py` uploads `model_source.py`
and runs it on the box, carrying R2 creds over SSH **stdin** (never argv). See its
`prefetch_models(...)` and `python -m tools.cloud_prefetch --host ... --port ... --model REPO@REV`.
Pair it with the `vast-gpu-rent` skill for the rent → prefetch → serve flow.

Full detail (bucket layout, upload path, cost): [references/usage.md](references/usage.md)
and `docs/guides/model-storage-r2.md`.

## Don't
- Don't commit R2 creds or `HF_TOKEN`; keep them in gitignored `.env`, and on a remote box
  pass them over SSH stdin only.
- Don't add a new model/revision to R2 without updating the bucket `MANIFEST.json` and the
  pinned repo+revision in code — the pull verifies against both.
- Don't use this for API models (qwen/gemini/luna) — they have no local weights.

## Guardrails
- Export `R2_*` before running (`set -a; . ./.env; set +a`). `model_source` does not read
  `.env`; without the vars it quietly uses HF. Confirm the log says `ready from R2`.
- `boto3` must be importable or the R2 client silently falls back to HF. It is pinned in
  `tools/videojev_probe/requirements.txt`; install with `python -m pip` (the venv's `pip`
  script can carry a stale shebang if the repo moved).
- The storage guide records VideoJev, CLIP, SigLIP2 and d3-nano as uploaded;
  Qwen is not yet mirrored. Use explicit repo+revision for d3-nano.
- HF fallback on large weights can hit the Xet CDN — set `HF_HUB_DISABLE_XET=1` for it.
- A damaged local R2 snapshot fails verification instead of being passed to HF's
  cache fast path. Inspect it and move that revision directory aside before retrying.
- Interrupted R2 attempts are discarded before HF fallback; they do not publish a
  partial HF snapshot. Fetching a pinned revision does not change HF `refs/main`.
- The read-write upload key is maintainers-only; the repo/CI uses the read-only token.
