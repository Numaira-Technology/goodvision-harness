"""Audit customer-facing release files and image tags, not image internals.

This is a repository-side release gate, not a customer runtime utility. It reads
only the allowlisted UTF-8 documents and the archive's manifest.json payload.
Image-layer contents and original third-party license notices are outside its
scope and must not be removed to satisfy this check.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import stat
import tarfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SCOPE = "customer-facing files and image tags"
TEXT_FILES = (
    "RELEASE.md",
    "FULL_DEPLOY.md",
    "API_DELIVERY.md",
    "EVENT_EVIDENCE.md",
    "compose.full-gpu.delivery.yaml",
    ".env.full-gpu.delivery.example",
    "config.delivery.yaml",
    "schemas/analysis-result-v1.schema.json",
    "schemas/analysis-result-v2.schema.json",
    "schemas/event-v1.schema.json",
    "tools/check_runtime.py",
    "tools/check_face.py",
    "tools/check_perception.py",
)
ARCHIVE_NAME = "goodvision-full-2026.09.03-gpu.1.tar.gz"
EXPECTED_IMAGE_TAGS = frozenset(
    {
        "goodvision-harness:2026.09.03-full-gpu.1",
        "goodvision-rich-label-worker:2026.09.03-full-gpu.1",
    }
)
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
REPOSITORY_TEXT_FILES = (
    "RELEASE.md", "DEPLOY.md", "API_DELIVERY.md",
    "docs/architecture/event-evidence.md",
    "compose.full-gpu.delivery.yaml", "compose.full-gpu.accelerated.yaml",
    ".env.full-gpu.delivery.example", "config.delivery.yaml",
    "schemas/analysis-result-v1.schema.json",
    "schemas/analysis-result-v2.schema.json", "schemas/event-v1.schema.json",
    "src/goodvision_harness/dashboard.html",
    "src/goodvision_harness/fleet_console.js",
    "src/goodvision_harness/camera_display.js",
    "src/goodvision_harness/camera_display_dense.js",
    "tools/customer_check.py",
)


def _load_shared_privacy_checker() -> Callable[[object], bool]:
    path = Path(__file__).resolve().parents[1] / "src/goodvision_harness/privacy.py"
    spec = importlib.util.spec_from_file_location("_delivery_privacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("The shared customer-privacy checker is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.contains_private_implementation_detail


_SHARED_CONTAINS_PRIVATE_DETAIL = _load_shared_privacy_checker()
# Repository/package hosting URLs and additional historical model aliases are
# delivery-specific; the shared helper covers runtime vendors and libraries.
_DELIVERY_ONLY_MARKERS = re.compile(
    r"(?<![a-z0-9])(?:pypi|pythonhosted|github|gitlab|bitbucket|"
    r"scrfd|retinaface|arcface|cosface|blackwell|geforce|"
    r"pp[\s_-]*(?:human|shitu))[\w.+:/-]*",
    re.IGNORECASE,
)


def _contains_private_detail(value: str) -> bool:
    return bool(
        _SHARED_CONTAINS_PRIVATE_DETAIL(value)
        or _DELIVERY_ONLY_MARKERS.search(value)
    )


def _is_link_or_reparse_point(metadata: Any) -> bool:
    return bool(
        stat.S_ISLNK(metadata.st_mode)
        or getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _issue(
    report: dict[str, Any], code: str, path: str, *, line: int | None = None
) -> None:
    issue: dict[str, Any] = {"code": code, "path": path}
    if line is not None:
        issue["line"] = line
    report["issues"].append(issue)


def _inventory(
    root: Path, expected: set[str], report: dict[str, Any]
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    allowed_directories = {"schemas", "tools"}

    def visit(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError:
            relative = directory.relative_to(root).as_posix()
            _issue(report, "directory_unreadable", relative)
            return
        for entry in entries:
            relative = entry.relative_to(root).as_posix()
            if _contains_private_detail(relative):
                _issue(report, "private_marker_in_name", relative)
            try:
                metadata = entry.lstat()
            except OSError:
                _issue(report, "entry_unreadable", relative)
                continue
            if _is_link_or_reparse_point(metadata):
                _issue(report, "symbolic_link_not_allowed", relative)
                continue
            if stat.S_ISDIR(metadata.st_mode):
                if relative in allowed_directories:
                    visit(entry)
                else:
                    # Reject unrelated folders without reading their contents.
                    _issue(report, "unexpected_directory", relative)
            elif stat.S_ISREG(metadata.st_mode):
                files[relative] = entry
                if relative not in expected:
                    _issue(report, "unexpected_file", relative)
            else:
                _issue(report, "unsupported_entry_type", relative)

    visit(root)
    for missing in sorted(expected - files.keys()):
        _issue(report, "missing_file", missing)
    return files


def _check_archive(path: Path, report: dict[str, Any]) -> None:
    report["image_tag_check"] = {"status": "failed"}
    try:
        # Read the compressed archive once, retaining only the bounded manifest.
        # Other payloads are skipped by the stream reader; layers are never
        # extracted or loaded into memory. Continue after the first manifest so
        # a later duplicate cannot pass the release gate.
        payload: bytes | None = None
        with tarfile.open(path, mode="r|*") as archive:
            manifest_count = 0
            for member in archive:
                if member.name != "manifest.json":
                    continue
                manifest_count += 1
                if manifest_count != 1:
                    _issue(report, "archive_manifest_count", ARCHIVE_NAME)
                    return
                if not member.isfile():
                    _issue(report, "archive_manifest_not_regular", ARCHIVE_NAME)
                    return
                if member.size > MAX_MANIFEST_BYTES:
                    _issue(report, "archive_manifest_too_large", ARCHIVE_NAME)
                    return
                source = archive.extractfile(member)
                if source is None:
                    _issue(report, "archive_manifest_unreadable", ARCHIVE_NAME)
                    return
                with source:
                    payload = source.read(MAX_MANIFEST_BYTES + 1)
                if len(payload) > MAX_MANIFEST_BYTES:
                    _issue(report, "archive_manifest_too_large", ARCHIVE_NAME)
                    return
            if manifest_count != 1 or payload is None:
                _issue(report, "archive_manifest_count", ARCHIVE_NAME)
                return
        entries = json.loads(payload.decode("utf-8"))
    except (OSError, EOFError, tarfile.TarError, UnicodeDecodeError, ValueError):
        # Never echo manifest values, layer paths, or underlying error text.
        _issue(report, "archive_manifest_invalid", ARCHIVE_NAME)
        return

    if not isinstance(entries, list) or len(entries) != 2:
        _issue(report, "archive_image_count", ARCHIVE_NAME)
        return
    tags: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            _issue(report, "archive_image_entry_invalid", ARCHIVE_NAME)
            return
        repo_tags = entry.get("RepoTags")
        if (
            not isinstance(repo_tags, list)
            or len(repo_tags) != 1
            or not isinstance(repo_tags[0], str)
        ):
            _issue(report, "archive_image_tags_invalid", ARCHIVE_NAME)
            return
        tags.extend(repo_tags)
    if set(tags) != EXPECTED_IMAGE_TAGS:
        _issue(report, "archive_image_tags_mismatch", ARCHIVE_NAME)
        return
    report["image_tag_check"] = {
        "status": "passed",
        "tags": sorted(EXPECTED_IMAGE_TAGS),
    }


def audit_bundle(
    directory: str | Path, *, without_archive: bool = False
) -> dict[str, Any]:
    """Return a bounded report; no package/driver inventory or byte checksums."""
    report: dict[str, Any] = {
        "ok": False,
        "scope": SCOPE,
        "mode": "text-only" if without_archive else "release",
        "expected_text_files": len(TEXT_FILES),
        "checked_text_files": 0,
        "image_tag_check": {"status": "not_checked"},
        "limitations": [
            "Image-layer contents and runtime behavior are not inspected.",
            "Underlying dependencies and original license notices are outside "
            "this audit; passing does not claim that they are absent.",
        ],
        "issues": [],
    }
    root = Path(directory).absolute()
    try:
        metadata = root.lstat()
    except OSError:
        _issue(report, "bundle_directory_unavailable", ".")
        return report
    if _is_link_or_reparse_point(metadata):
        _issue(report, "symbolic_link_not_allowed", ".")
        return report
    if not stat.S_ISDIR(metadata.st_mode):
        _issue(report, "bundle_path_not_directory", ".")
        return report

    expected = set(TEXT_FILES)
    if not without_archive:
        expected.add(ARCHIVE_NAME)
    files = _inventory(root, expected, report)
    for relative in TEXT_FILES:
        if _contains_private_detail(relative):
            _issue(report, "private_marker_in_allowlist", relative)
        path = files.get(relative)
        if path is None:
            continue
        try:
            metadata = path.lstat()
            if _is_link_or_reparse_point(metadata):
                _issue(report, "symbolic_link_not_allowed", relative)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                _issue(report, "unsupported_entry_type", relative)
                continue
            if metadata.st_size > MAX_TEXT_BYTES:
                _issue(report, "text_file_too_large", relative)
                continue
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            _issue(report, "text_not_utf8", relative)
            continue
        except OSError:
            _issue(report, "text_unreadable", relative)
            continue
        report["checked_text_files"] += 1
        if _contains_private_detail(text):
            first_line = next(
                (
                    number
                    for number, line in enumerate(text.splitlines(), start=1)
                    if _contains_private_detail(line)
                ),
                None,
            )
            _issue(report, "private_marker_in_text", relative, line=first_line)

    if not without_archive and ARCHIVE_NAME in files:
        _check_archive(files[ARCHIVE_NAME], report)
    report["ok"] = not report["issues"]
    return report


def audit_repository(directory: str | Path) -> dict[str, Any]:
    """Check current publication sources separately from historical bundles."""
    root = Path(directory)
    report: dict[str, Any] = {
        "ok": False,
        "scope": "current customer documents, console assets and delivery config",
        "mode": "repository",
        "checked_text_files": 0,
        "issues": [],
        "exceptions": [],
        "limitations": [
            "Runtime responses require the privacy and service regression tests.",
            "Internal source, historical deployment profiles, image layers and "
            "third-party runtime/license files are outside this publication scan.",
        ],
    }
    for relative in REPOSITORY_TEXT_FILES:
        try:
            content = (root / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            _issue(report, "text_unreadable", relative)
            continue
        report["checked_text_files"] += 1
        for number, line in enumerate(content.splitlines(), 1):
            if not _contains_private_detail(line):
                continue
            if (relative == "compose.full-gpu.accelerated.yaml"
                    and line.strip() == "- driver: nvidia"):
                # This is a required device-driver selector, not a product label.
                report["exceptions"].append({
                    "code": "required_gpu_driver_selector",
                    "path": relative, "line": number,
                })
                continue
            _issue(report, "private_marker_in_text", relative, line=number)
    report["ok"] = not report["issues"]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Customer release directory")
    parser.add_argument(
        "--repository", action="store_true",
        help="Check current publication source files instead of a release archive",
    )
    parser.add_argument(
        "--without-archive",
        action="store_true",
        help="Require the allowlisted text files before the image tar is exported",
    )
    args = parser.parse_args(argv)
    report = (
        audit_repository(args.directory) if args.repository
        else audit_bundle(args.directory, without_archive=args.without_archive)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
