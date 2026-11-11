#!/usr/bin/env python3
"""Derive vLLM plans from a verified HF bundle, retaining its seal and bytes."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from tools import registry
from tools.unified_benchmark.adapters import request_body
from tools.unified_benchmark.contract import (
    RULES,
    VERSION,
    digest,
    file_hash,
    write_new,
)
from tools.unified_benchmark.preflight import model_configs
from tools.unified_benchmark.workflow import verify_bundle

SEAL = "bundle-files-sha256.json"
HOST = "qwen8b_r2_host"
VLLM = "qwen8b_r2_vllm"
LINEAGE = "provenance/vllm-bundle-conversion.json"
SOURCE_SEAL = "provenance/source-bundle-files-sha256.json"


def _inventory(bundle):
    files = set()
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError("bundle conversion requires regular files, not links")
        if path.is_file():
            files.add(path.relative_to(bundle).as_posix())
        elif not path.is_dir():
            raise ValueError("bundle contains a special file")
    return files - {SEAL}


def _plans(bundle, alias, config, hashes):
    config_name = f"configs/{alias}.json"
    if (
        config_name not in hashes
        or json.loads((bundle / config_name).read_text()) != config
    ):
        raise ValueError("source profile differs from the reviewed configuration")
    plans = {}
    for split in ("validation", "test"):
        name = f"plans/{alias}-{split}.json"
        if name not in hashes:
            raise ValueError("plan is not sealed: " + name)
        plan = json.loads((bundle / name).read_text())
        if (
            plan.get("protocol") != VERSION
            or plan.get("split") != split
            or plan.get("config_sha256") != digest(config)
            or plan.get("rules_sha256") != digest(RULES)
        ):
            raise ValueError("plan identity differs: " + name)
        units = plan.get("units") or []
        ids = {unit["id"] for unit in units}
        if not units or len(ids) != len(units):
            raise ValueError("plan requires unique request units")
        for unit in units:
            for frame in unit["frames"]:
                if hashes.get(frame["path"]) != frame["sha256"]:
                    raise ValueError("request frame is not sealed")
            if digest(request_body(config, unit, bundle)[1]) != unit["request_sha256"]:
                raise ValueError("request fingerprint differs")
        if not plan.get("questions") or any(
            not question.get("unit_ids") or not set(question["unit_ids"]) <= ids
            for question in plan["questions"]
        ):
            raise ValueError("question references missing request units")
        plans[split] = plan
    return plans


def prepare(src: Path, dest: Path) -> dict:
    src, dest = Path(src), Path(dest)
    if dest.exists() or dest.is_symlink():
        raise ValueError(f"destination exists: {dest}")
    src, dest = src.resolve(), dest.resolve()
    if dest.is_relative_to(src):
        raise ValueError("destination must be outside the source bundle")
    inventory = _inventory(src)
    source_digest = verify_bundle(src)
    seal = (src / SEAL).read_bytes()
    hashes = json.loads(seal)
    if inventory != set(hashes):
        raise ValueError("source inventory contains unsealed or missing files")
    configs = model_configs([HOST, VLLM])
    plans = _plans(src, HOST, configs[HOST], hashes)
    additions = {f"configs/{VLLM}.json", LINEAGE, SOURCE_SEAL} | {
        f"plans/{VLLM}-{split}.json" for split in plans
    }
    if additions & inventory:
        raise ValueError("source already contains conversion outputs")
    env = registry.env_of("qwen8b-vllm")
    identity = {key: env[key] for key in ("r2_key", "sha256", "size")}
    if configs[VLLM].get("env_artifact") != identity:
        raise ValueError("reviewed vLLM profile and registry artifact differ")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vllm-bundle-", dir=dest.parent) as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()
        # Preserve original hashes: changed bytes must never gain a new seal.
        for name, expected in hashes.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / name, path, follow_symlinks=False)
            if path.is_symlink() or file_hash(path) != expected:
                raise ValueError("source changed during conversion: " + name)
        write_new(stage / f"configs/{VLLM}.json", configs[VLLM])
        for split, plan in plans.items():
            write_new(
                stage / f"plans/{VLLM}-{split}.json",
                {**plan, "config_sha256": digest(configs[VLLM])},
            )
        lineage = {
            "source_bundle_sha256": source_digest,
            "source_seal": SOURCE_SEAL,
            "converter_sha256": file_hash(__file__),
            "source_profile_sha256": digest(configs[HOST]),
            "derived_profile_sha256": digest(configs[VLLM]),
            "exam_unchanged": True,
            "scope": "same source questions/labels/frames/request bodies; "
            "new serving profile",
        }
        write_new(stage / LINEAGE, lineage)
        (stage / SOURCE_SEAL).write_bytes(seal)
        final_hashes = {
            **hashes,
            **{name: file_hash(stage / name) for name in additions},
        }
        write_new(stage / SEAL, final_hashes)
        verify_bundle(stage)
        _plans(stage, VLLM, configs[VLLM], final_hashes)
        if (src / SEAL).read_bytes() != seal or verify_bundle(src) != source_digest:
            raise ValueError("source seal changed during conversion")
        # copytree reserves a new directory on all supported platforms. Keep the
        # seal absent until every copied byte is verified; partial outputs fail
        # closed and are retained for inspection, never overwritten on retry.
        shutil.copytree(
            stage,
            dest,
            ignore=lambda directory, names: [SEAL] if Path(directory) == stage else [],
        )
        for name, expected in final_hashes.items():
            if file_hash(dest / name) != expected:
                raise ValueError("derived output changed during copying: " + name)
        write_new(dest / SEAL, final_hashes)
    return {
        "destination": str(dest),
        "config_sha256": digest(configs[VLLM]),
        "units_copied": sum(len(plan["units"]) for plan in plans.values()),
        "bundle_sha256": file_hash(dest / SEAL),
        "source_bundle_sha256": source_digest,
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
