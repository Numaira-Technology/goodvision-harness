"""Create a portable source/sample bundle for an already rented GPU instance."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def checked_file(path, root):
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"Refusing missing, linked or external source: {path}")
    return path


def source_files(root):
    tracked = (
        subprocess.check_output(
            ["git", "ls-files", "-z", "src/goodvision_harness"],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    files = [root / name for name in tracked if name]
    for directory, patterns in {
        "workers/sam_scene": ["*.py", "requirements.txt", "README.md"],
        "tools/seg_benchmark": ["*.py", "*.html", "requirements.txt"],
        "tools/vast_benchmark": ["*.py", "*.sh", "*.txt", "README.md"],
        "tools/seg_pipeline": ["*.py", "*.sh"],
        "tools/cloud_cpu_tests": ["*.py", "*.sh", "*.json", "README.md"],
    }.items():
        for pattern in patterns:
            files.extend((root / directory).glob(pattern))
    files.extend(
        root / name
        for name in (
            "pyproject.toml",
            "config.yaml",
            "README.md",
            "tools/export_seg_benchmark.py",
            "tools/run_seg_benchmark.sh",
            "tools/setup_seg_benchmark.sh",
            "tools/sam31_latency_run.py",
            "tools/sam31_session_run.py",
            "tools/sam31_video_run.py",
            "tools/evaluate_pipeline_latency.cjs",
            "docs/guides/segmentation-benchmark.md",
            "docs/guides/segmentation-pipeline-benchmark.md",
            "docs/validation/evidence/sam-display-latency-20260922.json",
        )
    )
    return sorted(set(files))


def stage_dataset(source, stage):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    labels = json.loads((source / "annotations.json").read_text())
    if labels["dataset_sha256"] != sha256(manifest_path):
        raise ValueError("Dataset and annotation identity differ")
    target = stage / "outputs/seg-benchmark/dataset"
    for frame in manifest["frames"]:
        relative = Path(frame["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid dataset image path")
        image = checked_file(source / relative, source)
        if sha256(image) != frame["sha256"]:
            raise ValueError(f"Frame checksum differs: {image}")
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, target / relative)
    clips = []
    for index, item in enumerate(manifest["sources"]):
        video = Path(item["path"])
        if not video.is_absolute():
            video = source / video
        if video.is_symlink() or not video.is_file():
            raise ValueError(f"Missing/linked original video: {video}")
        if sha256(video) != item["sha256"]:
            raise ValueError(f"Video checksum differs: {video}")
        filename = f"clip-{index + 1:02d}{video.suffix.lower()}"
        destination = stage / "samples/videos" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(video, destination)
        item["path"] = f"../../../samples/videos/{filename}"
        clips.append(
            {
                **item.get("provenance", {}),
                "id": item["id"],
                "path": filename,
                "sha256": item["sha256"],
            }
        )
    write_json(stage / "samples/videos/manifest.json", {"clips": clips})
    write_json(target / "manifest.json", manifest)
    labels["dataset_sha256"] = sha256(target / "manifest.json")
    write_json(target / "annotations.json", labels)
    template = (stage / "tools/seg_benchmark/annotate.html").read_text()
    data = json.dumps(
        {"manifest": manifest, "dataset_sha256": labels["dataset_sha256"]},
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    (target / "annotate.html").write_text(template.replace("__DATASET_JSON__", data))


def worker_defaults(root):
    tree = ast.parse((root / "tools/start_sam.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "WORKER_ENV" for t in node.targets
        ):
            # Extract individual constants without executing the rental controller.
            result = {}
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                name = ast.literal_eval(key)
                if name.startswith("SAM") and name != "SAM31_CHECKPOINT_PATH":
                    result[name] = ast.literal_eval(value)
            return result
    raise ValueError("Repository worker defaults not found")


def export(root, dataset, output, *, pipeline_video=None, meeting_tests=False):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="goodvision-bundle-") as temporary:
        stage = Path(temporary) / "goodvision-4090-benchmark"
        stage.mkdir()
        for path in source_files(root):
            checked_file(path, root)
            relative = path.relative_to(root)
            if any(
                part.startswith(".env")
                or part in {".git", ".goodvision", "__pycache__"}
                for part in relative.parts
            ) or path.suffix in {".pem", ".key", ".pt", ".pth"}:
                raise ValueError(f"Forbidden source path: {relative}")
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        write_json(
            stage / "tools/vast_benchmark/repository-worker-env.json",
            worker_defaults(root),
        )
        shutil.copyfile(
            stage
            / (
                "tools/cloud_cpu_tests/README.md"
                if meeting_tests
                else "tools/vast_benchmark/README.md"
            ),
            stage / "START_HERE.md",
        )
        if dataset is not None:
            stage_dataset(dataset, stage)
        if pipeline_video is not None:
            expected = json.loads(
                (
                    root / "docs/validation/evidence/sam-display-latency-20260922.json"
                ).read_text()
            )["input"]["sha256"]
            if sha256(pipeline_video) != expected:
                raise ValueError("Pipeline video differs from the SAM baseline")
            target = stage / "video_samples/sample_video.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pipeline_video, target)
        if dataset is None and pipeline_video is None:
            raise ValueError("At least one dataset/video is required")
        files = {
            str(p.relative_to(stage)): sha256(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        write_json(
            stage / "BUNDLE_MANIFEST.json",
            {
                "schema": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root
                )
                .decode()
                .strip(),
                "working_tree_snapshot": True,
                "gpu_runtime_verified": False,
                "purpose": "sam_cloud_cpu_tests"
                if meeting_tests
                else "segmentation_comparison",
                "files": files,
            },
        )
        partial = output.with_suffix(output.suffix + ".partial")
        with tarfile.open(partial, "w:gz") as archive:
            archive.add(stage, arcname=stage.name)
        partial.replace(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "outputs/seg-benchmark/dataset"
    )
    parser.add_argument("--pipeline-video", type=Path)
    parser.add_argument("--skip-offline-dataset", action="store_true")
    parser.add_argument("--meeting-tests", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / f"goodvision-4090-benchmark-{stamp}.tar.gz",
    )
    args = parser.parse_args()
    print(
        export(
            ROOT,
            None if args.skip_offline_dataset else args.dataset.resolve(),
            args.output.resolve(),
            pipeline_video=args.pipeline_video,
            meeting_tests=args.meeting_tests,
        )
    )


if __name__ == "__main__":
    main()
