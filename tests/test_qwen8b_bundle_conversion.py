"""Derived bundle conversion never repairs or reseals corrupted source evidence."""

import json

import pytest
from PIL import Image

from tools import registry, start_qwen8b_vllm
from tools.qwen8b import prepare_vllm_bundle as converter
from tools.unified_benchmark.adapters import request_body
from tools.unified_benchmark.contract import RULES, VERSION, digest, file_hash
from tools.unified_benchmark.preflight import freeze_sources, model_configs


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def seal(root):
    manifest = root / converter.SEAL
    manifest.write_text(
        json.dumps(
            {
                p.relative_to(root).as_posix(): file_hash(p)
                for p in root.rglob("*")
                if p.is_file() and p != manifest
            }
        )
    )


@pytest.fixture
def host_bundle(tmp_path):
    root = tmp_path / "host"
    for name in ("frames", "plans", "configs"):
        (root / name).mkdir(parents=True)
    config = model_configs([converter.HOST])[converter.HOST]
    (root / f"configs/{converter.HOST}.json").write_text(json.dumps(config))
    frame = root / "frames/synthetic.jpg"
    Image.new("RGB", (720, 404), (20, 40, 60)).save(frame)
    for split in ("validation", "test"):
        unit = {
            "id": f"{split}/0",
            "source_id": split,
            "targets": ["fight"],
            "question": "Is fighting visible?",
            "frames": [
                {"path": "frames/synthetic.jpg", "sha256": file_hash(frame), "time": 0}
            ],
        }
        unit["request_sha256"] = digest(request_body(config, unit, root)[1])
        plan = {
            "protocol": VERSION,
            "split": split,
            "config_sha256": digest(config),
            "rules_sha256": digest(RULES),
            "units": [unit],
            "questions": [
                {
                    "number": 1,
                    "event": "fight",
                    "reference": "no",
                    "unit_ids": [unit["id"]],
                }
            ],
        }
        (root / f"plans/{converter.HOST}-{split}.json").write_text(json.dumps(plan))
    (root / "rules.json").write_text(json.dumps(RULES))
    sources = freeze_sources(root)
    (root / "preflight.json").write_text(
        json.dumps(
            {
                "offline_passed": True,
                "sources_sha256": digest(sources),
                "models": {
                    alias: {
                        split: {
                            "plan_sha256": digest(
                                json.loads(
                                    (root / f"plans/{alias}-{split}.json").read_text()
                                )
                            )
                        }
                        for split in ("validation", "test")
                    }
                    for alias in ("qwen8b_r2_host",)
                },
            }
        )
    )
    seal(root)
    return root


def test_valid_conversion_retains_source_bytes_and_request_identities(
    host_bundle, tmp_path
):
    before = snapshot(host_bundle)
    dest = tmp_path / "derived"
    result = converter.prepare(host_bundle, dest)
    assert snapshot(host_bundle) == before
    for name, data in before.items():
        if name != converter.SEAL:
            assert (dest / name).read_bytes() == data
    assert (dest / converter.SOURCE_SEAL).read_bytes() == before[converter.SEAL]
    assert result["exam_unchanged"] is True
    assert result["source_bundle_sha256"] == file_hash(host_bundle / converter.SEAL)
    lineage = json.loads((dest / converter.LINEAGE).read_text())
    assert lineage["source_bundle_sha256"] == result["source_bundle_sha256"]
    for split in ("validation", "test"):
        old = json.loads(
            (host_bundle / f"plans/{converter.HOST}-{split}.json").read_text()
        )
        new = json.loads((dest / f"plans/{converter.VLLM}-{split}.json").read_text())
        assert {k: v for k, v in new.items() if k != "config_sha256"} == {
            k: v for k, v in old.items() if k != "config_sha256"
        }
    assert (
        start_qwen8b_vllm._verify_eval_bundle(
            dest, "both", registry.env_of("qwen8b-vllm")
        )
        == result["bundle_sha256"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "label",
        "frame",
        "config",
        "source",
        "missing-seal",
        "extra",
        "unsealed-plan",
        "request-resealed",
        "profile-resealed",
        "link",
    ],
)
def test_conversion_rejects_corrupted_or_unsealed_source(host_bundle, tmp_path, fault):
    plan_path = host_bundle / f"plans/{converter.HOST}-test.json"
    if fault in {"label", "request-resealed"}:
        plan = json.loads(plan_path.read_text())
        if fault == "label":
            plan["questions"][0]["reference"] = "yes"
        else:
            plan["units"][0]["request_sha256"] = "0" * 64
        plan_path.write_text(json.dumps(plan))
        if fault.endswith("resealed"):
            seal(host_bundle)
    elif fault in {"config", "profile-resealed"}:
        path = host_bundle / f"configs/{converter.HOST}.json"
        config = json.loads(path.read_text())
        config["max_tokens"] = 1
        path.write_text(json.dumps(config))
        if fault.endswith("resealed"):
            seal(host_bundle)
    elif fault == "frame":
        (host_bundle / "frames/synthetic.jpg").write_bytes(b"changed image")
    elif fault == "source":
        source = next((host_bundle / "source-snapshot").rglob("*.py"))
        source.write_text("# changed retained source")
    elif fault == "missing-seal":
        (host_bundle / converter.SEAL).unlink()
    elif fault == "extra":
        (host_bundle / "unsealed.txt").write_text("do not copy")
    elif fault == "unsealed-plan":
        path = host_bundle / converter.SEAL
        hashes = json.loads(path.read_text())
        del hashes[plan_path.relative_to(host_bundle).as_posix()]
        path.write_text(json.dumps(hashes))
    else:
        (host_bundle / "linked").symlink_to("frames")
    before = snapshot(host_bundle)
    dest = tmp_path / "rejected"
    with pytest.raises((ValueError, FileNotFoundError)):
        converter.prepare(host_bundle, dest)
    assert not dest.exists() and snapshot(host_bundle) == before


def test_conversion_never_creates_output_inside_source(host_bundle):
    before = snapshot(host_bundle)
    with pytest.raises(ValueError, match="outside the source"):
        converter.prepare(host_bundle, host_bundle / "derived")
    assert snapshot(host_bundle) == before


def test_copy_race_cannot_bless_changed_bytes(host_bundle, tmp_path, monkeypatch):
    real_copy = converter.shutil.copyfile

    def changed_copy(src, dest, **kwargs):
        result = real_copy(src, dest, **kwargs)
        if str(src).endswith(f"{converter.HOST}-test.json"):
            dest.write_bytes(b"tampered while copying")
        return result

    before = snapshot(host_bundle)
    monkeypatch.setattr(converter.shutil, "copyfile", changed_copy)
    with pytest.raises(ValueError, match="changed during conversion"):
        converter.prepare(host_bundle, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    assert snapshot(host_bundle) == before


@pytest.mark.parametrize(
    "fault", ["derived_plan", "source_inventory", "profile_lineage"]
)
def test_derived_preflight_requires_exact_lineage_bindings(
    host_bundle, tmp_path, fault
):
    from tools.unified_benchmark.workflow import verify_bundle

    dest = tmp_path / "derived"
    converter.prepare(host_bundle, dest)
    original_preflight = (dest / "preflight.json").read_bytes()
    if fault == "derived_plan":
        path = dest / f"plans/{converter.VLLM}-test.json"
        data = json.loads(path.read_text())
        data["questions"][0]["reference"] = "yes"
    elif fault == "source_inventory":
        path = dest / converter.SOURCE_SEAL
        data = json.loads(path.read_text())
        del data[f"plans/{converter.HOST}-test.json"]
    else:
        path = dest / converter.LINEAGE
        data = json.loads(path.read_text())
        data["derived_profile_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    if fault == "source_inventory":
        path = dest / converter.LINEAGE
        data = json.loads(path.read_text())
        data["source_bundle_sha256"] = file_hash(dest / converter.SOURCE_SEAL)
        path.write_text(json.dumps(data))
    seal(dest)
    with pytest.raises(ValueError):
        verify_bundle(dest)
    assert (dest / "preflight.json").read_bytes() == original_preflight
