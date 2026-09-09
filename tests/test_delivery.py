from __future__ import annotations

import importlib.util
import io
import json
import re
import stat
import tarfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

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


def test_text_only_audit_accepts_exact_ten_neutral_files(bundle: Path) -> None:
    report = audit.audit_bundle(bundle, without_archive=True)
    assert report["ok"] is True
    assert report["scope"] == "customer-facing files and image tags"
    assert report["expected_text_files"] == report["checked_text_files"] == 10
    assert report["image_tag_check"] == {"status": "not_checked"}
    assert any("not claim" in item for item in report["limitations"])


def test_release_audit_accepts_exact_tags_without_scanning_image_layers(
    bundle: Path,
) -> None:
    _write_archive(bundle)
    report = audit.audit_bundle(bundle)
    assert report["ok"] is True
    assert report["checked_text_files"] == 10
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


def _load_delivery() -> dict[str, Any]:
    return yaml.safe_load(DELIVERY_PATH.read_text(encoding="utf-8"))


def test_full_delivery_keeps_both_images_and_requires_gpu_profile_for_rich() -> None:
    services = _load_delivery()["services"]
    assert set(services) == {"harness", "rich-label-worker"}
    assert services["harness"]["image"] == (
        "goodvision-harness:2026.09.09-full.1"
    )
    assert services["rich-label-worker"]["image"] == (
        "goodvision-rich-label-worker:2026.09.09-full.1"
    )
    for service in services.values():
        assert "build" not in service
        assert service["pull_policy"] == "never"
    assert "profiles" not in services["harness"]
    assert services["rich-label-worker"]["profiles"] == ["gpu"]


def test_full_gpu_settings_are_fixed_despite_old_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RICH_LABEL_ENABLED", "false")
    monkeypatch.setenv("RICH_LABEL_ENDPOINT", "http://old-worker:8091/v1/labels")
    monkeypatch.setenv("FACE_PROVIDER", "CPUExecutionProvider")
    monkeypatch.setenv("HARNESS_INFERENCE_PROFILE", "development")
    monkeypatch.setenv("INFERENCE_PROVIDER", "aiclub")
    monkeypatch.setenv("INFERENCE_MODEL", "old-model")
    monkeypatch.setenv("INFERENCE_ENDPOINT", "http://old-service/v1/chat/completions")

    environment = _load_delivery()["services"]["harness"]["environment"]
    # Literal values cannot be substituted by old .env or shell settings.
    assert environment["RICH_LABEL_ENABLED"] == "false"
    assert environment["RICH_LABEL_ENDPOINT"] == (
        "http://rich-label-worker:8091/v1/labels"
    )
    assert environment["FACE_ACCELERATOR"] == "cpu"
    accelerated = yaml.safe_load(
        (ROOT / "compose.full-gpu.accelerated.yaml").read_text(encoding="utf-8")
    )["services"]["harness"]["environment"]
    assert accelerated["FACE_ACCELERATOR"] == "gpu"
    assert accelerated["RICH_LABEL_ENABLED"] == "true"
    assert "FACE_PROVIDER" not in environment
    assert environment["HARNESS_INFERENCE_PROFILE"] == "customer-standard"
    assert not any(
        key in environment
        for key in ("INFERENCE_PROVIDER", "INFERENCE_MODEL", "INFERENCE_ENDPOINT")
    )


def test_full_gpu_delivery_preserves_project_and_persistent_volumes() -> None:
    delivery = _load_delivery()
    assert delivery["name"] == "goodvision-harness"
    assert delivery["volumes"] == {
        "face-models": None,
        "face-data": None,
        "rich-models": None,
    }
    services = delivery["services"]
    assert "face-models:/models" in services["harness"]["volumes"]
    assert "face-data:/data" in services["harness"]["volumes"]
    assert "./config.delivery.yaml:/app/config.yaml:ro" in (
        services["harness"]["volumes"]
    )
    assert services["rich-label-worker"]["volumes"] == ["rich-models:/models"]


def test_full_gpu_worker_does_not_publish_host_ports() -> None:
    worker = _load_delivery()["services"]["rich-label-worker"]
    assert "ports" not in worker
    assert worker.get("network_mode") != "host"


def test_full_delivery_reserves_devices_only_in_gpu_overlay() -> None:
    for service in _load_delivery()["services"].values():
        assert "deploy" not in service
    accelerated = yaml.safe_load(
        (ROOT / "compose.full-gpu.accelerated.yaml").read_text(encoding="utf-8")
    )
    for service in accelerated["services"].values():
        devices = service["deploy"]["resources"]["reservations"]["devices"]
        assert devices == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]


def test_full_gpu_delivery_uses_neutral_credentials_and_model_profiles() -> None:
    services = _load_delivery()["services"]
    harness = services["harness"]["environment"]
    assert harness["INFERENCE_API_KEY"] == "${INFERENCE_API_KEY:-}"
    assert harness["HARNESS_INFERENCE_PROFILE"] == "customer-standard"
    assert not any(
        key in harness
        for key in ("AICLUB_API_KEY", "OPENROUTER_API_KEY", "DASHSCOPE_API_KEY")
    )
    assert harness["VIDEO_CAPTURE_OPTIONS"] == (
        "${VIDEO_CAPTURE_OPTIONS:-rtsp_transport;tcp|timeout;5000000}"
    )
    worker = services["rich-label-worker"]["environment"]
    assert worker["RICH_DEVICE"] == "gpu:0"
    assert worker["RICH_DETECTOR_PROFILE"] == "${RICH_DETECTOR_PROFILE:-accurate}"
    assert worker["RICH_ATTRIBUTE_PROFILE"] == "${RICH_ATTRIBUTE_PROFILE:-standard}"
    assert "RICH_DETECTOR_MODEL" not in worker
    assert "RICH_ATTRIBUTE_MODEL" not in worker
    for key in ("RICH_DETECTOR_MODEL_DIR", "RICH_ATTRIBUTE_MODEL_DIR"):
        assert worker[key] == "${" + key + ":-}"
    example = (ROOT / ".env.full-gpu.delivery.example").read_text(encoding="utf-8")
    assert not re.search(
        r"(?m)^(?:INFERENCE_PROVIDER|INFERENCE_MODEL|INFERENCE_ENDPOINT)=", example
    )


def test_neutral_delivery_configuration_preserves_analysis_settings() -> None:
    source = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    delivery = yaml.safe_load(
        (ROOT / "config.delivery.yaml").read_text(encoding="utf-8")
    )
    expected = deepcopy(source)
    expected["inference_service"].pop("provider")
    expected["inference_service"].pop("providers")
    expected["face"].pop("model_name")
    expected["face"].pop("provider")
    expected["face"].update(
        model_profile="standard", model_root="/models", accelerator="gpu"
    )
    assert delivery == expected


@pytest.mark.parametrize(
    "filename",
    [
        "API_DELIVERY.md",
        "DEPLOY.md",
        "compose.full-gpu.delivery.yaml",
        ".env.full-gpu.delivery.example",
        "config.delivery.yaml",
    ],
)
def test_customer_delivery_text_uses_neutral_implementation_names(
    filename: str,
) -> None:
    text = (ROOT / filename).read_text(encoding="utf-8")
    forbidden = re.compile(
        r"openrouter|qwen|insightface|paddle|picodet|buffalo|onnx|opencv|ffmpeg|"
        r"aiclub|nvidia|cuda|cudnn|tensorrt|huggingface|openai|anthropic|"
        r"deepseek|gemini|claude|pp[-_]yolo|pp[-_]lcnet",
        re.IGNORECASE,
    )
    assert forbidden.search(text) is None


def test_delivery_guide_only_references_neutral_runtime_check_entrypoints() -> None:
    text = (ROOT / "DEPLOY.md").read_text(encoding="utf-8")
    assert set(re.findall(r"tools/[a-z_]+\.py", text)) == {
        "tools/check_runtime.py",
        "tools/check_face.py",
        "tools/check_perception.py",
    }


def test_project_metadata_uses_a_neutral_inline_description() -> None:
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = project["project"]["readme"]
    assert isinstance(readme, dict)
    assert readme["content-type"] == "text/markdown"
    assert "text" in readme and "file" not in readme
    assert "GoodVision" in readme["text"]
    assert "license and attribution" in readme["text"]
    assert re.search(
        r"openrouter|qwen|insightface|paddle|aiclub|cuda|onnx|opencv",
        readme["text"],
        re.IGNORECASE,
    ) is None
