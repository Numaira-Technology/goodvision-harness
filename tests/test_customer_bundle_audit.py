from __future__ import annotations

import importlib.util
import io
import json
import stat
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
DELIVERY_PATH = ROOT / "compose.full-gpu.delivery.yaml"
SCRIPT_PATH = ROOT / "tools/audit_customer_bundle.py"
SPEC = importlib.util.spec_from_file_location("customer_bundle_audit", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    directory = tmp_path / "release"
    directory.mkdir()
    for relative in audit.TEXT_FILES:
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "GoodVision Linux Docker GPU CPU JSON HTTP OpenAPI\n"
            "python tools/check_runtime.py\n",
            encoding="utf-8",
        )
    return directory


def _write_archive(
    directory: Path,
    *,
    entries: Any = None,
    duplicate_manifest: bool = False,
    manifest_link: bool = False,
) -> None:
    if entries is None:
        entries = [{"RepoTags": [tag]} for tag in sorted(audit.EXPECTED_IMAGE_TAGS)]
    payload = json.dumps(entries).encode("utf-8")
    with tarfile.open(directory / audit.ARCHIVE_NAME, "w:gz") as archive:
        # This content is deliberately private and must not be inspected by a
        # customer-file gate. The real release keeps original runtime notices.
        layer = b"Paddle InsightFace and original license notices inside a layer"
        layer_info = tarfile.TarInfo("runtime-layer.tar")
        layer_info.size = len(layer)
        archive.addfile(layer_info, io.BytesIO(layer))
        for _ in range(2 if duplicate_manifest else 1):
            info = tarfile.TarInfo("manifest.json")
            if manifest_link:
                info.type = tarfile.SYMTYPE
                info.linkname = "runtime-layer.tar"
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def _codes(report: dict[str, Any]) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


def test_text_only_audit_accepts_exact_release_neutral_files(bundle: Path) -> None:
    report = audit.audit_bundle(bundle, without_archive=True)
    assert report["ok"] is True
    assert report["scope"] == "customer-facing files and image tags"
    assert report["expected_text_files"] == report["checked_text_files"] == 13
    assert report["image_tag_check"] == {"status": "not_checked"}
    assert any("not claim" in item for item in report["limitations"])


def test_release_audit_accepts_exact_tags_without_scanning_image_layers(
    bundle: Path,
) -> None:
    _write_archive(bundle)
    report = audit.audit_bundle(bundle)
    assert report["ok"] is True
    assert report["checked_text_files"] == 13
    assert report["image_tag_check"] == {
        "status": "passed",
        "tags": sorted(audit.EXPECTED_IMAGE_TAGS),
    }


def test_audit_rejects_missing_file(bundle: Path) -> None:
    (bundle / "API_DELIVERY.md").unlink()
    report = audit.audit_bundle(bundle, without_archive=True)
    assert report["ok"] is False
    assert {"code": "missing_file", "path": "API_DELIVERY.md"} in report["issues"]


def test_audit_rejects_extra_file(bundle: Path) -> None:
    (bundle / "notes.txt").write_text("GoodVision", encoding="utf-8")
    report = audit.audit_bundle(bundle, without_archive=True)
    assert "unexpected_file" in _codes(report)


def test_audit_rejects_extra_directory_without_reading_its_contents(
    bundle: Path,
) -> None:
    private_directory = bundle / "private"
    private_directory.mkdir()
    (private_directory / "notes.txt").write_text("Paddle", encoding="utf-8")
    report = audit.audit_bundle(bundle, without_archive=True)
    assert "unexpected_directory" in _codes(report)
    assert "private_marker_in_text" not in _codes(report)


def test_audit_rejects_brand_in_filename(bundle: Path) -> None:
    (bundle / "Paddle_notes.txt").write_text("GoodVision", encoding="utf-8")
    report = audit.audit_bundle(bundle, without_archive=True)
    assert {"unexpected_file", "private_marker_in_name"} <= _codes(report)


@pytest.mark.parametrize(
    "marker",
    [
        "OpenRouter",
        "qwen3.5-plus",
        "InsightFace",
        "PaddlePaddle",
        "PaddleX",
        "PaddleDetection",
        "PP-YOLOE_plus-L",
        "PP-LCNet_x1_0_pedestrian_attribute",
        "PP-Human",
        "PicoDet-S",
        "buffalo_l",
        "AIClub",
        "NVIDIA",
        "CUDAExecutionProvider",
        "cuDNN",
        "onnxruntime-gpu",
        "OpenCV",
        "FastAPI",
        "FFmpeg",
        "NumPy",
        "PyTorch",
        "https://aiclub.zimaocloud.com/model/openApi/v1/chat/completions",
        "https://openrouter.ai/api/v1",
        "https://paddle-model-ecology.bj.bcebos.com/models",
        "https://huggingface.co/example/model",
        "https://pypi.org/project/example",
        "https://files.pythonhosted.org/packages/example.whl",
        "https://github.com/example/project",
        "飞桨",
        "通义千问",
    ],
)
def test_audit_rejects_brand_or_model_in_text(bundle: Path, marker: str) -> None:
    (bundle / "RELEASE.md").write_text(
        f"GoodVision\n{marker}\n", encoding="utf-8"
    )
    report = audit.audit_bundle(bundle, without_archive=True)
    assert report["ok"] is False
    assert {
        "code": "private_marker_in_text",
        "path": "RELEASE.md",
        "line": 2,
    } in report["issues"]
    # Reports identify the file/line, not the private content.
    assert marker not in json.dumps(report)


def test_audit_rejects_non_utf8_text(bundle: Path) -> None:
    (bundle / "RELEASE.md").write_bytes(b"\xff\xfeinvalid")
    report = audit.audit_bundle(bundle, without_archive=True)
    assert "text_not_utf8" in _codes(report)


def test_audit_requires_archive_in_release_mode(bundle: Path) -> None:
    report = audit.audit_bundle(bundle)
    assert {"code": "missing_file", "path": audit.ARCHIVE_NAME} in report["issues"]


def test_text_only_mode_rejects_an_existing_archive_as_extra(bundle: Path) -> None:
    _write_archive(bundle)
    report = audit.audit_bundle(bundle, without_archive=True)
    assert "unexpected_file" in _codes(report)
    assert report["image_tag_check"]["status"] == "not_checked"


@pytest.mark.parametrize(
    "entries,code",
    [
        ([], "archive_image_count"),
        ({"RepoTags": []}, "archive_image_count"),
        ([None, None], "archive_image_entry_invalid"),
        ([{"RepoTags": []}, {"RepoTags": []}], "archive_image_tags_invalid"),
        (
            [{"RepoTags": ["old-version"]}, {"RepoTags": ["other-version"]}],
            "archive_image_tags_mismatch",
        ),
        (
            [
                {"RepoTags": ["goodvision-harness:2026.09.03-full-gpu.1"]},
                {"RepoTags": ["goodvision-harness:2026.09.03-full-gpu.1"]},
            ],
            "archive_image_tags_mismatch",
        ),
        (
            [
                {"RepoTags": sorted(audit.EXPECTED_IMAGE_TAGS)},
                {"RepoTags": ["extra-tag"]},
            ],
            "archive_image_tags_invalid",
        ),
    ],
)
def test_audit_rejects_inexact_image_manifest(
    bundle: Path, entries: Any, code: str
) -> None:
    _write_archive(bundle, entries=entries)
    report = audit.audit_bundle(bundle)
    assert code in _codes(report)
    assert report["image_tag_check"]["status"] == "failed"
    assert "old-version" not in json.dumps(report)


def test_audit_rejects_duplicate_manifest(bundle: Path) -> None:
    _write_archive(bundle, duplicate_manifest=True)
    report = audit.audit_bundle(bundle)
    assert "archive_manifest_count" in _codes(report)


def test_audit_rejects_linked_manifest(bundle: Path) -> None:
    _write_archive(bundle, manifest_link=True)
    report = audit.audit_bundle(bundle)
    assert "archive_manifest_not_regular" in _codes(report)


def test_audit_rejects_invalid_archive(bundle: Path) -> None:
    (bundle / audit.ARCHIVE_NAME).write_bytes(b"not an archive")
    report = audit.audit_bundle(bundle)
    assert "archive_manifest_invalid" in _codes(report)


def test_archive_audit_reads_only_manifest_payload(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_archive(bundle)
    original_extractfile = tarfile.TarFile.extractfile
    payload_reads: list[str] = []

    def tracked_extractfile(
        archive: tarfile.TarFile, member: tarfile.TarInfo
    ) -> Any:
        payload_reads.append(member.name)
        return original_extractfile(archive, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", tracked_extractfile)
    report = audit.audit_bundle(bundle)
    assert report["ok"] is True
    assert payload_reads == ["manifest.json"]


@pytest.mark.parametrize("relative", ["RELEASE.md", "tools", "."])
def test_link_or_reparse_metadata_is_rejected_on_every_platform(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    target = bundle / relative
    original_lstat = Path.lstat

    def simulated_link_lstat(path: Path) -> Any:
        if path == target:
            return SimpleNamespace(
                st_mode=stat.S_IFLNK if relative == "RELEASE.md" else stat.S_IFDIR,
                st_file_attributes=0x400,
            )
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", simulated_link_lstat)
    report = audit.audit_bundle(bundle, without_archive=True)
    assert report["ok"] is False
    assert "symbolic_link_not_allowed" in _codes(report)


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Symbolic links are unavailable in this test environment: {exc}")


def test_audit_rejects_file_symlink_without_reading_target(
    bundle: Path, tmp_path: Path
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("Paddle", encoding="utf-8")
    link = bundle / "RELEASE.md"
    link.unlink()
    _symlink_or_skip(link, target)
    report = audit.audit_bundle(bundle, without_archive=True)
    assert "symbolic_link_not_allowed" in _codes(report)
    assert "private_marker_in_text" not in _codes(report)


def test_audit_rejects_directory_symlink(bundle: Path, tmp_path: Path) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    _symlink_or_skip(bundle / "linked", target, directory=True)
    report = audit.audit_bundle(bundle, without_archive=True)
    assert "symbolic_link_not_allowed" in _codes(report)


def test_audit_rejects_root_symlink(bundle: Path, tmp_path: Path) -> None:
    link = tmp_path / "linked-release"
    _symlink_or_skip(link, bundle, directory=True)
    report = audit.audit_bundle(link, without_archive=True)
    assert report["issues"] == [{"code": "symbolic_link_not_allowed", "path": "."}]


def test_cli_returns_structured_success_report(
    bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert audit.main([str(bundle), "--without-archive"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["scope"] == audit.SCOPE


def test_cli_returns_nonzero_for_failed_gate(
    bundle: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert audit.main([str(bundle)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False


def test_repository_publication_audit_passes_with_only_required_driver_exceptions():
    report = audit.audit_repository(ROOT)
    assert report["ok"], report["issues"]
    assert report["checked_text_files"] == len(audit.REPOSITORY_TEXT_FILES)
    assert len(report["exceptions"]) == 2
    assert {item["code"] for item in report["exceptions"]} == {
        "required_gpu_driver_selector",
    }


@pytest.mark.parametrize("marker", [
    "SAM 3.1", "sam_live", "Segment Anything", "Reka", "Ultralytics",
])
def test_release_gate_rejects_new_tracking_implementation_markers(bundle, marker):
    (bundle / "API_DELIVERY.md").write_text(marker, encoding="utf-8")
    report = audit.audit_bundle(bundle, without_archive=True)
    assert not report["ok"]
    assert "private_marker_in_text" in _codes(report)
