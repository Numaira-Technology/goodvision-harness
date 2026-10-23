"""Portable source-only bundle and result export; never include credentials."""

import json
import tarfile
import tempfile
from pathlib import Path

from tools.vlm_benchmark.run_remote import bundle as harness_bundle
from tools.vlm_benchmark.run_remote import safe_extract

from .runtime import sha, write_json

ROOT = Path(__file__).resolve().parents[2]


def bundle(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("bundle destination exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        base = temporary / "base.tar.gz"
        harness_bundle(ROOT, [], base, scope="vast_self_hosted_qwen35")
        stage = temporary / "stage"
        stage.mkdir()
        safe_extract(base, stage)
        manifest = json.loads((stage / "bundle-manifest.json").read_text())
        sources = list((ROOT / "tools/vast_vlm").glob("*.py"))
        sources += list((ROOT / "tools/tracked_grounding").glob("*.py"))
        # Shared offline Astra comparison helper used by tracked reports.
        sources += list((ROOT / "tools/decisions_benchmark").glob("*.py"))
        sources += list((ROOT / "workers/detector_rich_label").glob("*.py"))
        sources += [
            ROOT / "tools/vast_vlm/setup-yolo.sh",
            ROOT / "docs/guides/vast-fp8-yolo.md",
            ROOT / "docs/guides/pr48-49-person-grounding-integration.md",
            ROOT / "tools/vast_vlm/Dockerfile",
            ROOT / "tools/vast_vlm/setup-client.sh",
            ROOT / "tools/vast_vlm/onstart-ssh.sh",
            ROOT / "docs/guides/vast-self-hosted-qwen35.md",
        ]
        for source in sources:
            if source.is_symlink():
                raise ValueError("bundle source cannot be a symlink")
            relative = source.relative_to(ROOT)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            manifest["files"][relative.as_posix()] = sha(target)
        write_json(stage / "bundle-manifest.json", manifest)
        with tarfile.open(destination, "x:gz") as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(stage), recursive=False)
    return {
        "bundle": str(destination),
        "sha256": sha(destination),
        "files": len(manifest["files"]),
    }


def collect(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (
        not (source / "manifest.json").is_file()
        or not (source / "summary.json").is_file()
    ):
        raise ValueError("source must be a benchmark results directory")
    if destination.is_relative_to(source) or destination.exists():
        raise ValueError("archive must be new and outside the result directory")
    files = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("result exports cannot contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part.startswith(".") for part in relative.parts):
            continue
        # Explicit result types, not arbitrary files or credentials from the volume.
        if path.suffix.lower() in {
            ".json",
            ".jsonl",
            ".html",
            ".csv",
            ".mp4",
            ".png",
            ".jpg",
        }:
            files.append((path, relative))
    hashes = {str(relative): sha(path) for path, relative in files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "x:gz") as archive:
        for path, relative in files:
            archive.add(path, arcname=relative, recursive=False)
    write_json(
        destination.with_suffix(".manifest.json"),
        {
            "archive_sha256": sha(destination),
            "files": hashes,
        },
    )
    return {
        "archive": str(destination),
        "files": len(files),
        "sha256": sha(destination),
    }
