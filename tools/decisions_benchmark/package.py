"""Export source only; credentials, video and reference stay outside the archive."""

import json
import tarfile
import tempfile
from pathlib import Path

from tools.vlm_benchmark.model_comparison import sha
from tools.vlm_benchmark.report import write_json
from tools.vlm_benchmark.run_remote import bundle as harness_bundle
from tools.vlm_benchmark.run_remote import safe_extract

ROOT = Path(__file__).resolve().parents[2]


def bundle(destination, *, tracked=False):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("bundle destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        base, stage = temporary / "base.tar.gz", temporary / "stage"
        harness_bundle(
            ROOT,
            [],
            base,
            scope="tracked_decisions" if tracked else "openai_decisions_events",
        )
        stage.mkdir()
        safe_extract(base, stage)
        manifest = json.loads((stage / "bundle-manifest.json").read_text())
        sources = [
            *sorted((ROOT / "tools/decisions_benchmark").glob("*.py")),
            ROOT / "docs/guides/decisions-api-benchmark.md",
        ]
        if tracked:
            sources += [
                *sorted((ROOT / "tools/tracked_decisions").glob("*.py")),
                *sorted((ROOT / "tools/tracked_grounding").glob("*.py")),
                *sorted((ROOT / "workers/detector_rich_label").glob("*.py")),
                ROOT / "workers/detector_rich_label/requirements.txt",
                ROOT / "docs/guides/yolo-decisions-benchmark.md",
            ]
        for source in sources:
            if source.is_symlink():
                raise ValueError("source symlinks are not allowed")
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
