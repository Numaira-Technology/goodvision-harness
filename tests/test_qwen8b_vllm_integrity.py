"""Offline artifact, bundle, smoke and host-selection regressions; no real services."""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tools import build_env, registry, start_qwen8b_vllm as launcher
from tools.diag_qwen8b_vllm_import import _eligible as diag_eligible
from tools.qwen8b import eval_local
from tools.unified_benchmark.adapters import request_body
from tools.unified_benchmark.contract import RULES, VERSION, digest, file_hash
from tools.unified_benchmark.preflight import freeze_sources, model_configs
from tools.unified_benchmark.retry import Ledger, Queue
from tools.unified_benchmark.workflow import verify_bundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real network/R2/model operation forbidden in this offline test")
    monkeypatch.setattr(build_env.model_source, "r2_client", forbidden)
    monkeypatch.setattr(eval_local.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(launcher, "restore_env", forbidden)
    monkeypatch.setattr(launcher, "prefetch_models", forbidden)
    monkeypatch.setattr(launcher.S, "create_from_offers", forbidden)


def seal(bundle):
    manifest = bundle / "bundle-files-sha256.json"
    manifest.write_text(json.dumps({str(p.relative_to(bundle)): file_hash(p)
        for p in bundle.rglob("*") if p.is_file() and p != manifest}))


@pytest.fixture
def frozen(tmp_path):
    """Synthetic request bundle, not real reviewed exam inputs or model evidence."""
    bundle = tmp_path / "bundle"
    for name in ("frames", "configs", "plans"):
        (bundle / name).mkdir(parents=True)
    config = model_configs(["qwen8b_r2_vllm"])["qwen8b_r2_vllm"]
    (bundle / "configs/qwen8b_r2_vllm.json").write_text(json.dumps(config))
    frame = bundle / "frames/synthetic.jpg"
    Image.new("RGB", (720, 404), (100, 20, 40)).save(frame)
    phases = {}
    for split in ("validation", "test"):
        unit = {"id": f"synthetic-{split}/0", "source_id": f"synthetic-{split}",
                "targets": ["fight"], "question": "Is fighting visible in the current image?",
                "frames": [{"path": "frames/synthetic.jpg", "sha256": file_hash(frame), "time": 0.0}]}
        unit["request_sha256"] = digest(request_body(config, unit, bundle)[1])
        plan = {"protocol": VERSION, "split": split, "config_sha256": digest(config),
                "rules_sha256": digest(RULES), "units": [unit],
                "questions": [{"number": 1, "event": "fight", "reference": "no", "unit_ids": [unit["id"]]}]}
        (bundle / f"plans/qwen8b_r2_vllm-{split}.json").write_text(json.dumps(plan))
        phases[split] = {"plan_sha256": digest(plan)}
    (bundle / "rules.json").write_text(json.dumps(RULES))
    sources = freeze_sources(bundle)
    (bundle / "preflight.json").write_text(json.dumps({
        "offline_passed": True,
        "sources_sha256": digest(sources),
        "models": {"qwen8b_r2_vllm": phases},
    }))
    seal(bundle)
    return bundle


@pytest.mark.parametrize("flags", [[], ["--pack-only"], ["--upload-only"]])
def test_published_ada2_stops_before_build_or_r2(tmp_path, monkeypatch, flags):
    def forbidden(*args, **kwargs):
        pytest.fail("frozen key reached build or pack")
    monkeypatch.setattr(build_env, "_ensure_miniconda", forbidden)
    monkeypatch.setattr(build_env, "_pack", forbidden)
    replacement = tmp_path / "replacement.tar.gz"
    replacement.write_bytes(b"not a real pack")
    with pytest.raises(SystemExit, match="refusing to upload over frozen"):
        build_env.main(["--model", "qwen8b-vllm", "--out", str(replacement), *flags])
    with pytest.raises(SystemExit, match="refusing to upload over frozen"):
        build_env._upload(
            registry.env_of("qwen8b-vllm"), "qwen8b-vllm", replacement, {},
            "synthetic", record={}, recipe_sha="0" * 64,
        )


def test_active_and_historical_pack_identities_are_not_conflated():
    env = registry.env_of("qwen8b-vllm")
    expected = {k: env[k] for k in ("r2_key", "sha256", "size")}
    config = model_configs(["qwen8b_r2_vllm"])["qwen8b_r2_vllm"]
    provenance = json.loads((ROOT / "tools/qwen8b_vllm/pack_provenance.json").read_text())
    historical = json.loads((ROOT / "tools/qwen8b_vllm/pack_provenance-cu130-ada-incomplete.json").read_text())
    assert config["env_artifact"] == expected
    assert {k: provenance[k] for k in expected} == expected
    assert provenance["verification"]["status"] == "declared_identity_not_independently_verified"
    assert provenance["verification"]["ada2_package_inventory"] is None
    assert historical["r2_key"] == launcher.INCOMPLETE_VLLM_KEY
    assert historical["sha256"] == env["incomplete_cu130"]["sha256"]
    assert historical["sha256"] != expected["sha256"]
    assert "NOT an ada2 lock" in (ROOT / "tools/qwen8b_vllm/requirements.lock").read_text()


def test_valid_bundle_is_checked_without_external_services(frozen):
    assert launcher._verify_eval_bundle(frozen, "both", registry.env_of("qwen8b-vllm")) == file_hash(frozen / "bundle-files-sha256.json")


@pytest.mark.parametrize("change", ["legacy", "key", "sha", "size", "tokens", "temperature", "weights", "engine"])
def test_even_resealed_stale_or_changed_profiles_fail(frozen, change):
    path = frozen / "configs/qwen8b_r2_vllm.json"
    cfg = json.loads(path.read_text())
    if change == "legacy":
        del cfg["env_artifact"]
        cfg["env_source"] = launcher.INCOMPLETE_VLLM_KEY
    elif change == "key":
        cfg["env_artifact"]["r2_key"] = launcher.INCOMPLETE_VLLM_KEY
    elif change == "sha":
        cfg["env_artifact"]["sha256"] = "0" * 64
    elif change == "size":
        cfg["env_artifact"]["size"] += 1
    else:
        field, value = {"tokens": ("max_tokens", 300), "temperature": ("temperature", 1),
                        "weights": ("weight_revision", "0" * 40), "engine": ("engine", "transformers")}[change]
        cfg[field] = value
    path.write_text(json.dumps(cfg))
    # Make a self-consistent synthetic bundle for the stale profile so this
    # tests the independent active-profile gate, not an earlier binding error.
    report_path = frozen / "preflight.json"
    report = json.loads(report_path.read_text())
    for split in ("validation", "test"):
        plan_path = frozen / f"plans/qwen8b_r2_vllm-{split}.json"
        plan = json.loads(plan_path.read_text())
        plan["config_sha256"] = digest(cfg)
        for unit in plan["units"]:
            unit["request_sha256"] = digest(request_body(cfg, unit, frozen)[1])
        plan_path.write_text(json.dumps(plan))
        report["models"]["qwen8b_r2_vllm"][split]["plan_sha256"] = digest(plan)
    report_path.write_text(json.dumps(report))
    seal(frozen)
    assert verify_bundle(frozen) == file_hash(
        frozen / "bundle-files-sha256.json"
    )
    before = {str(p.relative_to(frozen)): file_hash(p) for p in frozen.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="profile differs"):
        launcher._verify_eval_bundle(frozen, "both", registry.env_of("qwen8b-vllm"))
    assert before == {str(p.relative_to(frozen)): file_hash(p) for p in frozen.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["request", "config_hash", "rules_hash", "split", "unsealed", "empty", "duplicate"])
def test_changed_request_plan_fails_before_sending(frozen, change):
    path = frozen / "plans/qwen8b_r2_vllm-validation.json"
    plan = json.loads(path.read_text())
    if change == "request":
        plan["units"][0]["question"] = "A different question"
    elif change == "config_hash":
        plan["config_sha256"] = "0" * 64
    elif change == "rules_hash":
        plan["rules_sha256"] = "0" * 64
    elif change == "split":
        plan["split"] = "test"
    elif change == "empty":
        plan["units"] = []
    elif change == "duplicate":
        plan["units"] *= 2
    path.write_text(json.dumps(plan))
    # Bind the changed plan so request fingerprints and unique-unit guards are
    # still exercised instead of every case stopping at the plan digest check.
    report_path = frozen / "preflight.json"
    report = json.loads(report_path.read_text())
    report["models"]["qwen8b_r2_vllm"]["validation"]["plan_sha256"] = digest(plan)
    report_path.write_text(json.dumps(report))
    seal(frozen)
    if change == "unsealed":
        manifest = frozen / "bundle-files-sha256.json"
        hashes = json.loads(manifest.read_text())
        del hashes[str(path.relative_to(frozen))]
        manifest.write_text(json.dumps(hashes))
    expected = {
        "request": "request fingerprint differs",
        "config_hash": "preflight plan identity differs",
        "rules_hash": "preflight plan identity differs",
        "split": "preflight plan identity differs",
        "unsealed": "bundle inventory contains unsealed or missing files",
        "empty": "requires unique request units",
        "duplicate": "requires unique request units",
    }[change]
    with pytest.raises(ValueError, match=expected):
        launcher._verify_eval_bundle(frozen, "both", registry.env_of("qwen8b-vllm"))


@pytest.mark.parametrize("change", ["models", "plan_digest", "config", "frame"])
def test_resealing_does_not_replace_semantic_preflight_bindings(frozen, change):
    report_path = frozen / "preflight.json"
    report = json.loads(report_path.read_text())
    if change == "models":
        del report["models"]
    elif change == "plan_digest":
        report["models"]["qwen8b_r2_vllm"]["validation"]["plan_sha256"] = "0" * 64
    elif change == "config":
        path = frozen / "configs/qwen8b_r2_vllm.json"
        config = json.loads(path.read_text())
        config["max_tokens"] += 1
        path.write_text(json.dumps(config))
    else:
        (frozen / "frames/synthetic.jpg").write_bytes(b"changed synthetic frame")
    report_path.write_text(json.dumps(report))
    seal(frozen)
    expected = {
        "models": "missing preflight model/plan records",
        "plan_digest": "preflight plan identity differs",
        "config": "preflight plan identity differs",
        "frame": "unsealed bundle frame",
    }[change]
    with pytest.raises(ValueError, match=expected):
        launcher._verify_eval_bundle(frozen, "both", registry.env_of("qwen8b-vllm"))


def test_test_phase_preserves_declared_validation_plan(frozen):
    # Selecting test-only inference does not need validation for its smoke,
    # but the complete preflight inventory must remain intact.
    assert launcher._verify_eval_bundle(
        frozen, "test", registry.env_of("qwen8b-vllm")
    ) == file_hash(frozen / "bundle-files-sha256.json")
    (frozen / "plans/qwen8b_r2_vllm-validation.json").unlink()
    seal(frozen)
    with pytest.raises(ValueError, match="unsealed bundle plan"):
        launcher._verify_eval_bundle(frozen, "test", registry.env_of("qwen8b-vllm"))


def test_invalid_bundle_stops_main_before_credentials_or_rental(frozen, tmp_path, monkeypatch):
    (frozen / "configs/qwen8b_r2_vllm.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["launcher", "--eval-bundle", str(frozen), "--eval-output", str(tmp_path / "out")])
    monkeypatch.setattr(launcher.S, "required", lambda *_: pytest.fail("credentials read before verification"))
    with pytest.raises(ValueError, match="bundle file differs"):
        launcher.main()


def payload(content='{"answer":false}'):
    return {"model": launcher.MODEL_ID, "choices": [{"finish_reason": "stop", "message": {"content": content}}]}


def test_synthetic_smoke_never_consumes_scored_attempts(frozen, tmp_path, monkeypatch):
    cfg = json.loads((frozen / "configs/qwen8b_r2_vllm.json").read_text())
    unit = json.loads((frozen / "plans/qwen8b_r2_vllm-validation.json").read_text())["units"][0]
    requests = []
    class FakeOpener:
        def open(self, req, **kwargs):
            requests.append(json.loads(req.data))
            return io.BytesIO(json.dumps(payload() if len(requests) == 1 else payload("not JSON")).encode())
    opener = FakeOpener()
    receipt = tmp_path / "smoke.json"
    launcher._smoke(opener, receipt)
    record = json.loads(receipt.read_text())
    assert record["state"] == "completed" and record["included_in_evaluation"] is False
    assert record["request_sha256"] == digest(requests[0])
    assert record["response"] == payload()
    image_uri = requests[0]["messages"][0]["content"][1]["image_url"]["url"]
    with Image.open(io.BytesIO(base64.b64decode(image_uri.split(",", 1)[1]))) as image:
        image.load()  # Force decoding the exact transmitted JPEG, not only its header.
        assert image.format == "JPEG" and image.size == (720, 404)
    assert requests[0]["max_tokens"] == 30 and requests[0]["temperature"] == 0
    monkeypatch.setattr(eval_local.urllib.request, "urlopen", opener.open)
    ledger = Ledger(tmp_path / "attempts.jsonl", {"synthetic": True})
    try:
        transport = eval_local.LocalTransport(frozen, cfg, "http://127.0.0.1:1", tmp_path / "run", ledger)
        now = [0.0]
        queue = Queue(ledger, "choice", transport, clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n))
        queue.one(unit)
        queue.one(unit)  # An exhausted budget cannot be reset by continuation.
        assert len(requests) == 4
        assert digest(requests[0]) != unit["request_sha256"]
        assert all(digest(body) == unit["request_sha256"] for body in requests[1:])
        assert len([r for r in ledger.rows if r["type"] == "intent"]) == 3
    finally:
        ledger.close()


def test_existing_smoke_receipt_prevents_another_call(tmp_path):
    receipt = tmp_path / "smoke.json"
    receipt.write_text("retained evidence")
    opener = SimpleNamespace(open=lambda *a, **k: pytest.fail("duplicate diagnostic was sent"))
    with pytest.raises(FileExistsError):
        launcher._smoke(opener, receipt)
    assert receipt.read_text() == "retained evidence"


@pytest.mark.parametrize("response", [payload("not JSON"), {"choices": []}, payload() | {"model": "wrong"}])
def test_invalid_smoke_response_is_preserved_and_rejected(tmp_path, response):
    opener = SimpleNamespace(open=lambda *a, **k: io.BytesIO(json.dumps(response).encode()))
    receipt = tmp_path / "smoke.json"
    with pytest.raises(ValueError):
        launcher._smoke(opener, receipt)
    record = json.loads(receipt.read_text())
    assert record["state"] == "failed" and record["response"] == response


@pytest.mark.parametrize("raw", [b"not JSON", b'{"score":NaN}', b'{"score":Infinity}'])
def test_malformed_smoke_json_keeps_raw_failure_receipt(tmp_path, raw):
    opener = SimpleNamespace(open=lambda *a, **k: io.BytesIO(raw))
    receipt = tmp_path / "smoke.json"
    with pytest.raises(ValueError):
        launcher._smoke(opener, receipt)
    record = json.loads(receipt.read_text())
    assert record["state"] == "failed" and record["response_body"] == raw.decode()


def offer(**kwargs):
    return {"id": 1, "num_gpus": 1, "gpu_ram": 24564, "dph_total": 1.20,
            "cuda_max_good": 13.0, "driver_version": "580.65.06", **kwargs}


@pytest.mark.parametrize("changes", [{"cuda_max_good": 12.8}, {"driver_version": "570.124.06"},
    {"cuda_max_good": None}, {"cuda_max_good": float("nan")}, {"cuda_max_good": float("inf")},
    {"driver_version": "unknown"}, {"driver_version": None}])
def test_incompatible_or_unknown_hosts_cannot_be_rented(changes):
    bad = offer(**changes)
    with pytest.raises(RuntimeError, match="CUDA 13.0"):
        launcher._eligible_offers([bad])
    with pytest.raises(RuntimeError, match="CUDA 13.0"):
        diag_eligible([bad])


def test_cuda13_filters_preserve_hf_cuda12_requirements():
    assert "cuda_max_good>=13.0" in launcher._vast_query(80)
    assert "cuda_vers>=12.8" in launcher.H._vast_query(80)
    assert launcher._eligible_offers([offer(cuda_max_good=12.8), offer(id=2)]) == [offer(id=2)]
    assert diag_eligible([offer(id=2)]) == [offer(id=2)]


@pytest.mark.parametrize("output,valid", [("580.65.06\n", True), ("590.1\n", True),
    ("570.124.06\n", False), ("\n", False), ("unknown\n", False), ("580.65.06\n580.65.06\n", False)])
def test_actual_driver_is_checked_before_downloads(monkeypatch, output, valid):
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *a, **k: output)
    if valid:
        launcher._verify_host_driver(["synthetic-ssh"])
    else:
        with pytest.raises(RuntimeError, match="driver >=580"):
            launcher._verify_host_driver(["synthetic-ssh"])


def fake_main_runtime(monkeypatch, tmp_path, frozen, *, resume=None, mutate_after_boot=False):
    """Run real main control flow with every external service replaced by local fakes."""
    events = []
    key = tmp_path / "synthetic-ssh-key"
    key.write_text("synthetic test bytes; never used by SSH")
    args = ["launcher", "--eval-bundle", str(frozen), "--eval-output", str(tmp_path / "result")]
    if resume is not None:
        args += ["--resume-instance-id", "123"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(launcher, "_status", lambda *a: None)
    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda *a, **k: None))
    monkeypatch.setattr(launcher.S, "IMAGE", "synthetic-original-image")
    monkeypatch.setattr(launcher.S, "required", lambda name: str(key) if name == "SAM_SSH_KEY" else "synthetic-not-a-key")
    for name in launcher.H.R2_KEYS:
        monkeypatch.setenv(name, "synthetic-not-a-credential")
    def search(**kwargs):
        events.append("search")
        assert "cuda_max_good>=13.0" in kwargs["query"]
        return [offer(cuda_max_good=12.8), offer(id=2)]
    vast = SimpleNamespace(search_offers=search, show_instance=lambda **k: resume, show_instances=lambda: [])
    def client(**kwargs):
        events.append("client")
        return vast
    monkeypatch.setitem(sys.modules, "vastai", SimpleNamespace(VastAI=client))
    verify = launcher._verify_eval_bundle
    def preflight(*args):
        events.append("preflight")
        return verify(*args)
    monkeypatch.setattr(launcher, "_verify_eval_bundle", preflight)
    monkeypatch.setattr(launcher.S, "prepare_ssh_key", lambda path, dest: path)
    def create(_vast, chosen, **kwargs):
        events.append("rent")
        assert chosen == [offer(id=2)]
        return 123
    monkeypatch.setattr(launcher.S, "create_from_offers", create)
    monkeypatch.setattr(launcher.H, "attach_key", lambda *a: events.append("attach"))
    monkeypatch.setattr(launcher.H, "wait_for_ssh", lambda *a: ("synthetic.invalid", 22))
    def check_output(argv, **kwargs):
        if argv[-1] == "nvidia-smi --query-gpu=driver_version --format=csv,noheader":
            events.append("driver")
        return "580.65.06\n"
    monkeypatch.setattr(launcher.subprocess, "check_output", check_output)
    monkeypatch.setattr(launcher.S, "run", lambda *a, **k: events.append("bootstrap"))
    monkeypatch.setattr(launcher.H, "_upload_repo", lambda *a: events.append("repo"))
    monkeypatch.setattr(launcher, "prefetch_models", lambda **k: events.append("weights"))
    def restore(**kwargs):
        events.append("restore")
        return "/synthetic/bin/activate"
    monkeypatch.setattr(launcher, "restore_env", restore)
    monkeypatch.setattr(launcher, "_verify_restored_env", lambda *a: None)
    monkeypatch.setattr(launcher, "_remote_snapshot", lambda *a: "/synthetic/snapshot")
    worker = SimpleNamespace(_log_file=None)
    monkeypatch.setattr(launcher, "_server_process", lambda *a: worker)
    def ready(*args):
        if mutate_after_boot:
            (frozen / "configs/qwen8b_r2_vllm.json").write_text("{}")
    monkeypatch.setattr(launcher, "_wait_ready", ready)
    def open_response(req, **kwargs):
        events.append("smoke")
        return io.BytesIO(json.dumps(payload()).encode())
    monkeypatch.setattr(launcher, "build_opener", lambda *a: SimpleNamespace(open=open_response))
    def evaluate(argv):
        events.append("eval")
        assert "--execute" in argv and "qwen8b_r2_vllm" in argv
        return 0
    monkeypatch.setattr(eval_local, "main", evaluate)
    monkeypatch.setattr(launcher.S, "stop_process", lambda *a: events.append("stop"))
    monkeypatch.setattr(launcher.S, "release_instance", lambda *a: events.append("release"))
    return events


def test_main_checks_before_rental_downloads_and_inference(frozen, tmp_path, monkeypatch):
    events = fake_main_runtime(monkeypatch, tmp_path, frozen)
    assert launcher.main() == 0
    assert events.index("preflight") < events.index("client") < events.index("rent")
    assert events.index("driver") < events.index("bootstrap") < events.index("weights")
    assert events.count("preflight") == 2
    assert events[-5:] == ["preflight", "smoke", "eval", "stop", "release"]
    assert json.loads((tmp_path / "vllm-smoke-123.json").read_text())["included_in_evaluation"] is False


def test_bundle_changed_during_boot_stops_before_any_inference(frozen, tmp_path, monkeypatch):
    events = fake_main_runtime(monkeypatch, tmp_path, frozen, mutate_after_boot=True)
    assert launcher.main() == 1
    assert events.count("preflight") == 2
    assert "smoke" not in events and "eval" not in events
    assert "release" in events


@pytest.mark.parametrize("metadata", [offer(cuda_max_good=12.8), offer(driver_version="570.124.06"), {}])
def test_incompatible_resume_is_not_bootstrapped_or_destroyed(frozen, tmp_path, monkeypatch, metadata):
    events = fake_main_runtime(monkeypatch, tmp_path, frozen,
        resume=metadata | {"label": "goodvision-qwen8b-vllm-synthetic"})
    assert launcher.main() == 1
    assert not ({"rent", "attach", "bootstrap", "weights", "restore", "smoke", "eval", "release"} & set(events))


def test_compatible_resume_runs_driver_guard_without_renting(frozen, tmp_path, monkeypatch):
    events = fake_main_runtime(monkeypatch, tmp_path, frozen,
        resume=offer() | {"label": "goodvision-qwen8b-vllm-synthetic"})
    assert launcher.main() == 0
    assert "rent" not in events and "search" not in events
    assert events.index("driver") < events.index("weights")
    assert "release" in events
