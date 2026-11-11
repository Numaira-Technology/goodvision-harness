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

## Published artifact and evidence limits

The active registry/profile identity is:

- `envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz`
- SHA-256 `59314d00783b6698ef83cbfbd2e028c2f094d76c9d3383f5a37c3fdf3687b95b`
- Size: 4,252,375,781 bytes
- Side manifest: `envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.json`

`pack_provenance.json` records this **declared identity**, its source, and the
missing independent evidence. It does not assert a new GET, restore, GPU serve,
or exam. Obtain the retained readback/import/server logs, ada2 package inventory,
and frozen exam receipts/ledger before accepting those historical claims.

The old `pack_provenance-cu130-ada-incomplete.json` is preserved unchanged as a
historical record for the incomplete cu130 `ada` pack. `requirements.lock` is
that old pack's inventory, **not** an ada2 lock. It must not be relabelled or used
as evidence of ada2 contents. Both historical vLLM packs and the active ada2 key
are frozen against upload, as is the separate HF pack.

CUDA 13.x requires NVIDIA driver branch >=580. The vLLM rental query requires
CUDA >=13.0, filters returned driver/capability metadata, and checks the actual
driver before bootstrap or downloads. These requirements also apply to resumed
instances and the import diagnostic. The CUDA 12.8 container tag does not upgrade
the host driver. See [NVIDIA compatibility requirements](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
An incompatible or unidentified resumed instance is rejected before taking
ownership; it is left running/unmodified and remains the caller's responsibility.

## Build a successor (CPU VM pack → a new R2 key; GPU only restores)

Do **not** pip-install vLLM on the rented 4090. Pack CUDA wheels on a CPU
builder (`require_cuda: false` checks `torch.version.cuda`,
not `torch.cuda.is_available()`), upload a new reviewed R2 key, then the GPU launcher
`restore_env`s that tarball. Never upload over `envs/qwen8b/cu128-py312-torch280-ada.tar.gz`.

Before a rebuild, prepare a reviewed successor registry entry with distinct
tarball/manifest keys. The existing ada2 key cannot be rebuilt in place. The
following command only builds/packs locally; publishing a successor and updating
the launcher's fixed pins are separate reviewed steps.

```sh
sudo mkdir -p /opt/env && sudo chown "$USER" /opt/env
PYTHONPATH=. python -m tools.build_env \
  --model qwen8b-vllm \
  --out /tmp/qwen8b-vllm-successor.tar.gz --skip-upload
```

`env.build.install_default_torch: false` skips the HF env's `torch==2.8.0` pin so
`vllm==0.31.0` can pull its own torch (2.13.x in the in-repo lock).
`env.build.flash_attn: false` skips a second flash-attn compile; the vLLM wheel
ships its kernels. Preserve the new artifact's inventory, isolated restore and
readback evidence. Update registry, launcher pins, profile and provenance together
for an approved successor; create new evaluation bundles rather than editing old ones.

The default recipe is `tools/qwen8b_vllm/requirements.txt`. The builder checks the
declared Python/torch/CUDA/vLLM versions, required import paths, conda file inventory
and a fresh safe restore. Its version 2 `.build.json` binds the exact pack bytes
and recipe to those measured facts and explicitly records `gpu_validation: deferred`.
CPU checks do not establish GPU serving or ABI acceptance. Publishing requires a
reviewed successor entry and the matching producer record; uploads create new
content-addressed objects conditionally and never overwrite frozen keys.

To split building and packing, use `--skip-pack` with the intended `--out`, then
`--pack-only --skip-upload` with the same model, recipe and output. Keep the
generated `<out>.prefix.json`; packing verifies that record and probes the existing
prefix file/mode/link fingerprint before executing it again. The final record keeps
the original producer identity and source snapshot. Source RECORD mismatches require
matching original/installed hashes in declared conda relocation metadata; otherwise
they fail closed. After conda-unpack, original RECORD hash checks are explicitly
skipped because relocation rewrites files; missing-file and import checks remain.
This is not ABI acceptance. It does not reconstruct provenance for an arbitrary prefix.
An existing environment, tarball or record is preserved rather than overwritten.

Historical R2 keys (retained, refused). CUDA in the cu130 name matches the declared
torch wheel (`2.13.0+cu130`). Do not overwrite the first vLLM pack
`envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz` (sha256 `2a3dcb1f…`;
it omitted `typing_extensions` from the prefix) or the HF pack.

- `envs/qwen8b-vllm/cu130-py312-vllm0310-ada.tar.gz`
  sha256 `24f1eb0cc264a9794979eb6c2225eca47d67d68025407d51db86aa171ce7ba7e`
  size `4252947071` (historical declared readback; incomplete mixed-setuptools pack)
- `envs/qwen8b-vllm/cu130-py312-vllm0310-ada.json`

Pack with `PYTHONNOUSERSITE=1` / `python -s` so pip cannot satisfy deps
from `~/.local`. After conda-pack, restore into a fresh directory and
import the OpenAI API server and `torch.utils.cpp_extension` with `python -s`
before upload. Top-level torch/vLLM imports alone did not catch the old pack's
missing files. These local code changes do not establish real-pack acceptance.

Unpack prefix: `/opt/env/qwen8b-vllm` (not `/opt/env/qwen8b`).

## Weights

Reuse the existing R2 weight prefix (read-only, not rebuilt):

`models/Qwen/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/`
