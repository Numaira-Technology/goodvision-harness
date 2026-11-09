# qwen8b-vllm packed env (do not overwrite the HF qwen8b env)

This is a **second** R2 environment for Qwen3-VL-8B Instruct
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. The transformers+flash-attn pack
`envs/qwen8b/cu128-py312-torch280-ada.tar.gz` (sha256 `d101d333…`) stays the
record of the previous HF run. Never upload to that key.

## Restore the current package

For an ordinary evaluation, restore the existing environment instead of rebuilding
it. `tools/models.yaml` (`qwen8b-vllm.env`) and `tools.start_qwen8b_vllm` select:

| Field | Current verified package |
|---|---|
| R2 object | `envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz` |
| R2 manifest | `envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.json` |
| SHA-256 | `59314d00783b6698ef83cbfbd2e028c2f094d76c9d3383f5a37c3fdf3687b95b` |
| Size | `4252375781` bytes |
| Runtime | Python 3.12; torch 2.13.0+cu130; vLLM 0.31.0; Ada GPU |
| Unpack prefix | `/opt/env/qwen8b-vllm` |

Provide the private R2 credentials and authorized rental scope described in the
[Qwen run guide](../unified_benchmark/README.md#self-hosted-qwen8b-from-r2). The
launcher restores this object via `tools.env_source`, checks the pinned SHA-256
before unpacking (and size against the side-manifest when available), then
verifies CUDA and delayed server imports. Do not
substitute the older package or install missing dependencies on the GPU to bypass
a failure. The model weights remain a separate, pinned R2 object prefix below.

[PR #98](https://github.com/Numaira-Technology/goodvision-harness/pull/98) records
a full 4090 validation/test run with this `ada2` package. That is evidence for
that run, not an exemption from a new machine's driver, import and smoke checks.
The CUDA 12.8 base-image label does not describe this pack's CUDA 13.0 torch wheels.

## Historical packages (retained, not restore fallbacks)

- `envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz`: SHA-256
  `2a3dcb1fc6a8d84439263512c65dae0bb576c50d4f559ae9f069407eb2e86db6`,
  size `4236991211`; omitted `typing_extensions` from the prefix.
- `envs/qwen8b-vllm/cu130-py312-vllm0310-ada.tar.gz`: SHA-256
  `24f1eb0cc264a9794979eb6c2225eca47d67d68025407d51db86aa171ce7ba7e`,
  size `4252947071`; mixed setuptools sources and installation records, with
  missing delayed-import files. Frozen and refused by the launcher.

`pack_provenance.json` is the retained record for the second, incomplete object.
Its old digest and CPU-only checks must not be presented as verification of `ada2`.
Preserve that historical record rather than replacing it with a new filename.

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

## Build a future package (separate authorized maintenance)

Do **not** pip-install vLLM on the rented 4090. Pack CUDA wheels on a CPU
builder (`require_cuda: false` checks `torch.version.cuda`, not
`torch.cuda.is_available()`). Before using the build command, select fresh
versioned `r2_key` and `manifest_key` values in the registry and preserve all
existing objects, including `ada2` and the HF pack. The builder uploads to its
configured keys, so do not run it against the current registry for a normal rerun.

```sh
sudo mkdir -p /opt/env && sudo chown "$USER" /opt/env
PYTHONPATH=. python -m tools.build_env \
  --model qwen8b-vllm \
  --requirements tools/qwen8b_vllm/requirements.txt
```

`env.build.install_default_torch: false` skips the HF env's `torch==2.8.0` pin so
`vllm==0.31.0` can pull its own torch (2.13.x in the in-repo lock).
`env.build.flash_attn: false` skips a second flash-attn compile; the vLLM wheel
ships its kernels. A future package needs full-download SHA-256/size verification,
matching registry and launcher pins, an updated profile `env_source`, and matching
run instructions before it can be used. Do not update only the display name.

Pack with `PYTHONNOUSERSITE=1` / `python -s` so pip cannot satisfy deps
from `~/.local`. After conda-pack, restore into a fresh directory and
check the package's installation records, `vllm.entrypoints.openai.api_server`
and `torch.utils.cpp_extension` imports as well as torch/vllm. Missing files must
fail packing rather than be hidden by `--ignore-missing-files`. CPU validation
does not replace authorized GPU restore, smoke and evaluation for a future pack.

## Weights

Reuse the existing R2 weight prefix (read-only, not rebuilt):

`models/Qwen/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/`
