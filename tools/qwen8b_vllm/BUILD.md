# qwen8b-vllm packed env (do not overwrite the HF qwen8b env)

This is a **second** R2 environment for Qwen3-VL-8B Instruct
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. The transformers+flash-attn pack
`envs/qwen8b/cu128-py312-torch280-ada.tar.gz` (sha256 `d101d333…`) stays the
record of the previous HF run. Never upload to that key.

## Pins (official + this repo)

- Architecture: `Qwen3VLForConditionalGeneration` is on the vLLM supported-models
  list (`Qwen/Qwen3-VL-4B-Instruct, etc.`). Dense 8B Instruct is that family, not
  the 30B MoE (`Qwen3VLMoeForConditionalGeneration`).
- Engine: `vllm==0.31.0` — already used in `tools/vlm_benchmark/local_model_profile.json`
  for a Qwen3-VL family 4090-class profile. PyPI latest `0.33.0` wants `torch==2.14.0`
  / CUDA 13 extras; we do not chase that on the CUDA 12.8 4090 image without a
  measured install.
- Transformers: `5.14.1` (vLLM 0.31.0 lock in this repo). The HF packed env used
  `transformers==5.17.0`. Mark that as a serving-stack difference, not a JPEG/hash
  difference.
- FlashAttention: vLLM may select `FLASH_ATTN` (FA2 on Ada sm_89). That is an
  attention backend inside the engine, not a reason to skip vLLM.

## Build (CPU VM pack → R2; GPU only restores)

Do **not** pip-install vLLM on the rented 4090. Pack CUDA wheels on a CPU
builder (`require_cuda: false` checks the declared CUDA wheel versions without
claiming GPU execution). The published HF and cu128 vLLM keys below are frozen;
the builder refuses them even for a rebuild. Before building, review a successor
registry entry with a new key/manifest key and a fresh fixed unpack prefix.
Do not alter a historical bundle or manufacture evidence for an old pack.

Use a trusted installed Miniforge (or a verified local installer and digest).
Only after selecting that reviewed successor entry, build locally without upload:

```sh
sudo mkdir -p /opt/env && sudo chown "$USER" /opt/env
PYTHONPATH=. python -m tools.build_env \
  --model qwen8b-vllm \
  --requirements tools/qwen8b_vllm/requirements.txt \
  --out /private/build-output/qwen8b-successor.tar.gz \
  --skip-upload
```

`env.build.install_default_torch: false` skips the HF env's `torch==2.8.0` pin so
`vllm==0.31.0` can pull its own torch (2.13.x in the in-repo lock).
`env.build.flash_attn: false` skips a second flash-attn compile; the vLLM wheel
ships its kernels. The producer sidecar binds actual pack bytes, recipe, prefix
and recorded CPU import facts; GPU validation remains explicitly deferred.
`--skip-pack` and `--pack-only` additionally bind the staged prefix fingerprint.
An independently authorized `--upload-only` requires that trusted sidecar and
uses create-only content-addressed objects. Review the returned identity and
actual manifest/readback evidence before updating the successor's registry and
active profile. CPU import success does not prove CUDA kernels, model serving,
or the target GPU ABI; those require separate real-machine evidence.

Retained historical cu128 declaration (not a build/upload destination):

- `envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz`
- `envs/qwen8b-vllm/cu128-py312-vllm0310-ada.json`

#98 records this object as missing `typing_extensions` in the prefix. Keep the
pin frozen. Do not treat the ada2 exam as acceptance of this object, and do not
overwrite it. This statement is not a producer record or a restore log.

Unpack prefix: `/opt/env/qwen8b-vllm` (not `/opt/env/qwen8b`).

The serving launcher validates local declarations and sealed requests before
rental; that is not a remote object-availability check. Serving mode and phase
timeouts are not a whole-job cost or duration cap. Use a separately reviewed
bounded wrapper for an approved diagnostic; do not treat this guide as spending
authorization.

## Weights

Reuse the existing R2 weight prefix (read-only, not rebuilt):

`models/Qwen/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/`
