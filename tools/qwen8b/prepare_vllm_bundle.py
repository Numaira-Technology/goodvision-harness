#!/usr/bin/env python3
"""Copy a frozen HF host bundle and add qwen8b_r2_vllm plans.

Does not resample JPEGs, rewrite questions, or change request bytes. Unit
request_sha256 values stay those of the identical chat bodies. Updates
bundle-files-sha256.json so verify_bundle still passes.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tools.unified_benchmark.contract import digest, file_hash

PROFILES = Path(__file__).resolve().parents[1] / "unified_benchmark" / "profiles"


def prepare(src: Path, dest: Path) -> dict:
    src, dest = src.resolve(), dest.resolve()
    if dest.exists():
        raise SystemExit(f"destination exists: {dest}")
    shutil.copytree(src, dest, symlinks=False)
    config = json.loads((PROFILES / "qwen8b_r2_vllm.json").read_text())
    (dest / "configs" / "qwen8b_r2_vllm.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    cfg_digest = digest(config)
    copied = 0
    for split in ("validation", "test"):
        host = json.loads((dest / "plans" / f"qwen8b_r2_host-{split}.json").read_text())
        plan = json.loads(json.dumps(host))
        plan["config_sha256"] = cfg_digest
        (dest / "plans" / f"qwen8b_r2_vllm-{split}.json").write_text(
            json.dumps(plan, ensure_ascii=False, separators=(",", ":")) + "\n")
        copied += len(plan["units"])
    hashes = {str(p.relative_to(dest)): file_hash(p)
              for p in sorted(dest.rglob("*")) if p.is_file() and p.name != "bundle-files-sha256.json"}
    (dest / "bundle-files-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return {
        "destination": str(dest),
        "config_sha256": cfg_digest,
        "units_copied": copied,
        "bundle_sha256": file_hash(dest / "bundle-files-sha256.json"),
        "exam_unchanged": True,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    print(json.dumps(prepare(args.from_bundle, args.output)))


if __name__ == "__main__":
    raise SystemExit(main())
