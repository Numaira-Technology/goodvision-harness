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

from tools.unified_benchmark.adapters import request_body
from tools.unified_benchmark.contract import digest, file_hash
from tools.unified_benchmark.workflow import verify_bundle

PROFILES = Path(__file__).resolve().parents[1] / "unified_benchmark" / "profiles"


def prepare(src: Path, dest: Path) -> dict:
    src, dest = src.resolve(), dest.resolve()
    if dest.exists():
        raise SystemExit(f"destination exists: {dest}")
    if dest.is_relative_to(src):
        raise ValueError("derived bundle must be outside the source bundle")
    source_digest = verify_bundle(src)
    source_hashes = json.loads((src / "bundle-files-sha256.json").read_text())
    # Copy only the validated inventory, never unrelated files or symlink targets.
    dest.mkdir(parents=True)
    for name in source_hashes:
        path = src / name
        if path.is_symlink():
            raise ValueError("bundle derivation does not follow symlinks: " + name)
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    config = json.loads((PROFILES / "qwen8b_r2_vllm.json").read_text())
    if (dest / "configs" / "qwen8b_r2_vllm.json").exists():
        raise ValueError("source bundle already contains the vLLM variant")
    (dest / "configs" / "qwen8b_r2_vllm.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    cfg_digest = digest(config)
    copied = 0
    report = json.loads((dest / "preflight.json").read_text())
    counts = {}
    for split in ("validation", "test"):
        host = json.loads((dest / "plans" / f"qwen8b_r2_host-{split}.json").read_text())
        plan = json.loads(json.dumps(host))
        plan["config_sha256"] = cfg_digest
        for unit in plan["units"]:
            if digest(request_body(config, unit, dest)[1]) != unit["request_sha256"]:
                raise ValueError("vLLM profile changes a frozen request")
        (dest / "plans" / f"qwen8b_r2_vllm-{split}.json").write_text(
            json.dumps(plan, ensure_ascii=False, separators=(",", ":")) + "\n")
        copied += len(plan["units"])
        counts[split] = {"questions": len(plan["questions"]), "unique_units": len(plan["units"]),
                         "target_evaluations": sum(len(q["unit_ids"]) for q in plan["questions"]),
                         "plan_sha256": digest(plan)}
    report["models"]["qwen8b_r2_vllm"] = counts
    report["derived_from_bundle_sha256"] = source_digest
    report["derivation"] = "new vLLM variant; existing inputs, frames and request bytes retained"
    (dest / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    hashes = {p.relative_to(dest).as_posix(): file_hash(p)
              for p in sorted(dest.rglob("*")) if p.is_file() and p.name != "bundle-files-sha256.json"}
    (dest / "bundle-files-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    verify_bundle(dest)
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
