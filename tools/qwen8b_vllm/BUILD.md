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
builder (this cloud VM is fine: `require_cuda: false` checks `torch.version.cuda`,
not `torch.cuda.is_available()`), upload the new R2 key, then the GPU launcher
`restore_env`s that tarball. Never upload over `envs/qwen8b/cu128-py312-torch280-ada.tar.gz`.

```sh
sudo mkdir -p /opt/env && sudo chown "$USER" /opt/env
PYTHONPATH=. python -m tools.build_env \
  --model qwen8b-vllm \
  --requirements tools/qwen8b_vllm/requirements.txt
```

`env.build.install_default_torch: false` skips the HF env's `torch==2.8.0` pin so
`vllm==0.31.0` can pull its own torch (2.13.x in the in-repo lock).
`env.build.flash_attn: false` skips a second flash-attn compile; the vLLM wheel
ships its kernels. After upload, fill `env.sha256` / `env.size` in
`tools/models.yaml` so later 4090 boots skip pip.

R2 keys (versioned, new objects):

- `envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz`
- `envs/qwen8b-vllm/cu128-py312-vllm0310-ada.json`

Unpack prefix: `/opt/env/qwen8b-vllm` (not `/opt/env/qwen8b`).

## Weights

Reuse the existing R2 weight prefix (read-only, not rebuilt):

`models/Qwen/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/`
