# r2-model-pull — detail

Authoritative source for layout, cost and the upload path: `docs/guides/model-storage-r2.md`.
Code: `tools/model_source.py`, `tools/cloud_prefetch.py`.

## Pinned models in R2 (bucket `goodvision-models`)
Keyed `models/<org>/<name>/<40-char commit sha>/` with a per-revision `MANIFEST.json`.
- VideoJev-4B — `ZhuYuhan/VideoJev-4B` @ `1c1a253bf90dfa66176124d25a48ffb791d517d5` (~9 GB, includes the trust_remote_code `.py` files).
- CLIP — `openai/clip-vit-large-patch14` @ `32bd64288804d66eefd0ccbe215aa642df71cc41` (~1.7 GB).
The pinned repo+revision for `--model videojev|clip` come from code (`videojev_core.py`, `neural_service.py`), not this file.

## What `model_source.ensure_cached(repo, revision, cache_dir)` does
1. If a verified local snapshot already exists in `cache_dir`, use it (no network).
2. Else, with R2 creds present, pull from R2: download each file in `MANIFEST.json`,
   verify size + sha256 in a temporary staging directory, reject symlink/path-escape,
   then publish the complete snapshot. Existing HF branch refs are preserved.
3. Else fall back to `huggingface_hub.snapshot_download`.
An existing R2 snapshot with a malformed/mismatched manifest or damaged files fails
closed. Inspect and move that revision directory aside before retrying; HF must not
reuse files already known to be corrupt. Failed R2 staging directories are discarded.
The result fills a standard HF cache (`models--<org>--<name>/snapshots/<rev>/`), so any
`from_pretrained(..., cache_dir=..., local_files_only=True)` loads it offline.

## Verify a local pull
```bash
PYTHONPATH=src python -c "
from pathlib import Path
from tools import model_source as ms
import json
cache = Path('<hf-cache>')
for repo, rev in [('openai/clip-vit-large-patch14','32bd64288804d66eefd0ccbe215aa642df71cc41')]:
    snap = ms.snapshot_dir(cache, repo, rev)
    man = json.loads((snap / ms.MANIFEST_NAME).read_text()) if (snap/ms.MANIFEST_NAME).exists() else None
    print(repo, 'ok' if (man and ms.verify_snapshot(snap, man)) else 'present', snap.exists())
"
```

## Remote prefetch onto a rented GPU
```python
from tools.cloud_prefetch import prefetch_models
cache = prefetch_models(
    ssh_options=[...], host=HOST, port=PORT, remote_dir="/workspace/videojev",
    models=[("ZhuYuhan/VideoJev-4B","1c1a253bf90dfa66176124d25a48ffb791d517d5")],
    r2_env={k: os.environ[k] for k in ("R2_ENDPOINT","R2_BUCKET","R2_ACCESS_KEY_ID","R2_SECRET_ACCESS_KEY")},
    hf_token=os.environ.get("HF_TOKEN"),
)  # returns the remote HF cache dir; creds travel over SSH stdin, never argv
```
For the full rent → prefetch → serve flow, see the `vast-gpu-rent` skill.

## Cost (see the guide for current numbers)
R2 Standard storage + Class A/B ops, egress-free. Pulling the same revision repeatedly is
cheap; don't re-upload revisions needlessly (管理、不要 abuse).
