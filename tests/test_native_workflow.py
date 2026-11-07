"""Native CLI integration with synthetic receipts only: no model, GPU or network."""
import json
import threading
import time
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from tools.unified_benchmark import native_deploy as deploy, native_run as run, preflight, workflow
from tools.unified_benchmark.adapters import D3_OPTIONS, request_body
from tools.unified_benchmark.contract import EVENTS, RULES, digest, file_hash, write_new
from tools.unified_benchmark.retry import Ledger, Queue, PausedRun


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real network/model calls forbidden in native offline tests")
    monkeypatch.setattr(run.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(run.NativeBackend, "start", forbidden)


def seal(bundle):
    manifest = bundle / "bundle-files-sha256.json"
    manifest.write_text(json.dumps({str(p.relative_to(bundle)): file_hash(p)
        for p in bundle.rglob("*") if p.is_file() and p != manifest}))


@pytest.fixture
def frozen(tmp_path):
    bundle, deployment = tmp_path / "bundle", tmp_path / "deployment"
    bundle.mkdir()
    deployment.mkdir()
    frames = []
    for i in range(2):
        path = bundle / f"frame-{i}.jpg"
        Image.new("RGB", (8, 8), (10 + i, 20, 30)).save(path)
        frames.append({"path": path.name, "time": float(i), "sha256": file_hash(path)})
    files = {}
    for alias in deploy.ALIASES:
        cfg = deploy.profile(alias)
        write_new(bundle / f"configs/{alias}.json", cfg)
        for split, counts in (("validation", (6, 6, 6, 6, 0, 0)), ("test", (32, 22, 26, 20, 10, 10))):
            units, questions = [], []
            for event, count in zip(EVENTS, counts):
                for i in range(count):
                    answer = "yes" if i < count // 2 else "no"
                    uid = f"{split}/{event}/{i}/{answer}"
                    u = {"id": uid, "source_id": uid, "targets": [event],
                         "frames": frames[:cfg["window_frames"]], "question": None}
                    u["request_sha256"] = digest(request_body(cfg, u, bundle)[1])
                    units.append(u)
                    questions.append({"number": len(questions) + 1, "event": event,
                                      "reference": answer, "unit_ids": [uid]})
            write_new(bundle / f"plans/{alias}-{split}.json", {"split": split,
                "config_sha256": digest(cfg), "rules_sha256": digest(RULES), "units": units, "questions": questions})
        snapshot = deployment / alias / "snapshot"
        snapshot.mkdir(parents=True)
        weight = snapshot / "synthetic.safetensors"
        weight.write_bytes(b"synthetic fixture; never load as a model")
        path = deployment / f"checkpoints/{alias}.json"
        write_new(path, {"passed": True, "model": cfg["model"], "revision": cfg["revision"],
            "snapshot": str(snapshot), "files": [{"name": weight.name, "bytes": weight.stat().st_size,
                                                  "sha256": file_hash(weight)}]})
        files[str(path.relative_to(deployment))] = file_hash(path)
    write_new(deployment / "deployment.json", {"complete": True, "models": list(deploy.ALIASES), "files": files})
    write_new(bundle / "rules.json", RULES)
    sources = preflight.freeze_sources(bundle)
    write_new(bundle / "preflight.json", {"offline_passed": True, "sources_sha256": digest(sources)})
    seal(bundle)
    return bundle, deployment, file_hash(deployment / "deployment.json")


class FakeBackend:
    instances = []
    calls = []
    behavior = None

    def __init__(self, alias, cfg, bundle, deployment, output, port):
        self.alias, self.cfg, self.closed = alias, cfg, False
        self.instances.append(self)

    def start(self):
        pass

    def infer(self, unit, body):
        self.calls.append(unit["id"])
        # A synthetic answer encoded in the fixture ID, never a real model invocation.
        answer = unit["id"].rsplit("/", 1)[1]
        if type(self).behavior:
            override = type(self).behavior(self, unit, body)
            if override is not None:
                return override
        return 200, self.raw(unit, answer), {}

    def raw(self, unit, answer):
        if self.cfg["kind"] == "margin":
            margin = .3 if answer == "yes" else -.3
            return {e: {"margin": margin, "positive_cosine": .5 + margin, "contrast_cosine": .5}
                    for e in unit["targets"]}
        index = ("yes", "no", "unknown").index(answer)
        probabilities = [.05, .05, .05]
        probabilities[index] = .9
        if self.alias == "videojev":
            options = {q["id"]: q["options"] for q in self.cfg["question_template"]}
            raw = [{"id": "scene/" + e, "task": "choice", "answer_index": index,
                    "answer": options["scene/" + e][index], "probabilities": probabilities} for e in unit["targets"]]
        else:
            raw = {"answers": {e: {"type": "choice", "choice": D3_OPTIONS[index],
                "probabilities": dict(zip(D3_OPTIONS, probabilities))} for e in unit["targets"]}}
        return {"request_id": unit["id"], "model": self.cfg["model"], "revision": self.cfg["revision"], "raw": raw}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset_backend():
    FakeBackend.calls, FakeBackend.instances, FakeBackend.behavior = [], [], None


def immediate_queue(*args, **kwargs):
    tick = [time.time() + 1000000]
    return Queue(*args, **kwargs, clock=lambda: tick[0], sleep=lambda seconds: tick.__setitem__(0, tick[0] + seconds))


def execute(frozen, output, alias="clip", phase="both", **kwargs):
    bundle, deployment, sha = frozen
    run.execute(alias, bundle, deployment, output, sha, phase=phase,
        authorization="synthetic offline test", backend_factory=FakeBackend, queue_factory=immediate_queue, **kwargs)


@pytest.mark.parametrize("alias", deploy.ALIASES)
def test_four_native_validation_and_test_with_frozen_outputs(frozen, tmp_path, alias):
    out = tmp_path / "run"
    execute(frozen, out, alias)
    result = json.loads((out / "test/report.json").read_text())
    assert result["correct"] == result["accuracy_denominator"] == 120
    assert result["final_missing_unique_units"] == result["incomplete_questions"] == 0
    assert len(FakeBackend.calls) == 144 and all(b.closed for b in FakeBackend.instances)
    if alias in ("clip", "siglip2"):
        artifact = json.loads((out / "thresholds.json").read_text())
        assert artifact["validation_summary"]["complete_valid_questions"] == 24
        assert artifact["thresholds"] == {e: 0 for e in EVENTS}
        assert artifact["native_deployment_sha256"] == frozen[2]
    else:
        assert not (out / "thresholds.json").exists()
    manifest = json.loads((out / "result-files-sha256.json").read_text())
    assert "result-files-sha256.json" not in manifest
    assert all(file_hash(out / name) == sha for name, sha in manifest.items())
    before = {name: file_hash(out / name) for name in ("test/attempts.jsonl", "test/report.json")}
    execute(frozen, out, alias, resume=True)
    assert len(FakeBackend.calls) == 144 and len(FakeBackend.instances) == 1
    assert all(file_hash(out / name) == sha for name, sha in before.items())


def test_three_attempt_limit_missing_denominator_and_unknown_no_retry(frozen, tmp_path):
    def behavior(self, unit, body):
        if unit["id"] == "test/fight/0/yes": return 503, {"error": "synthetic transient"}, {}
        if unit["id"] == "test/fight/1/yes": return 200, self.raw(unit, "unknown"), {}
    FakeBackend.behavior = behavior
    out = tmp_path / "run"
    execute(frozen, out, "decision3", "test")
    counts = Counter(FakeBackend.calls)
    assert counts["test/fight/0/yes"] == 3 and counts["test/fight/1/yes"] == 1
    result = json.loads((out / "test/report.json").read_text())
    assert result["correct"] == 118 and result["accuracy_denominator"] == 120
    assert result["final_missing_unique_units"] == result["incomplete_questions"] == 1
    execute(frozen, out, "decision3", "test", resume=True)
    assert Counter(FakeBackend.calls) == counts  # exhausted units never get attempt 4


def test_incomplete_validation_blocks_test_and_keeps_partial_report(frozen, tmp_path):
    FakeBackend.behavior = lambda self, unit, body: (503, {"error": "fixture"}, {}) if unit["id"] == "validation/fight/0/yes" else None
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="incomplete validation"):
        execute(frozen, out)
    assert all(c.startswith("validation/") for c in FakeBackend.calls)
    assert not (out / "test").exists() and not (out / "thresholds.json").exists()
    assert list(out.glob("invocations/*/validation-partial-report.json"))


def test_resume_keeps_first_answers_and_cumulative_attempts(frozen, tmp_path):
    FakeBackend.behavior = lambda self, unit, body: (403, {"error": "fixture stop"}, {}) if unit["id"] == "test/fight/0/yes" else None
    out = tmp_path / "run"
    with pytest.raises(PausedRun): execute(frozen, out, "videojev", "test")
    ledger_before = (out / "test/attempts.jsonl").read_bytes()
    FakeBackend.behavior = None
    execute(frozen, out, "videojev", "test", resume=True)
    assert (out / "test/attempts.jsonl").read_bytes().startswith(ledger_before)
    assert Counter(FakeBackend.calls)["test/fight/0/yes"] == 2
    assert max(Counter(FakeBackend.calls).values()) == 2
    assert json.loads((out / "test/report.json").read_text())["correct"] == 120


def test_resume_after_validation_does_not_recalibrate_or_resend_it(frozen, tmp_path):
    FakeBackend.behavior = lambda self, unit, body: (403, {"error": "stop test"}, {}) if unit["id"].startswith("test/") else None
    out = tmp_path / "run"
    with pytest.raises(PausedRun): execute(frozen, out)
    frozen_threshold = file_hash(out / "thresholds.json")
    FakeBackend.behavior = None
    execute(frozen, out, resume=True)
    assert sum(c.startswith("validation/") for c in FakeBackend.calls) == 24
    assert file_hash(out / "thresholds.json") == frozen_threshold
    assert json.loads((out / "test/report.json").read_text())["correct"] == 120


def test_orphan_valid_receipt_recovered_without_another_request(frozen, tmp_path):
    bundle, _, _ = frozen
    cfg = deploy.profile("clip")
    plan = json.loads((bundle / "plans/clip-validation.json").read_text())
    unit = plan["units"][0]
    phase = tmp_path / "run/validation"
    ledger = Ledger(phase / "attempts.jsonl", workflow.context_for(plan, cfg, workflow.verify_bundle(bundle)))
    ledger.append({"unit": unit["id"], "attempt": 1, "type": "intent", "request_sha256": unit["request_sha256"]})
    raw = {"fight": {"margin": .3, "positive_cosine": .5, "contrast_cosine": .2}}
    from tools.unified_benchmark.adapters import parse_response
    parsed = parse_response(cfg, unit, raw)
    write_new(phase / "receipts" / f"{digest(unit['id'])}-1.json", {"unit": unit["id"], "attempt": 1,
        "request_sha256": unit["request_sha256"], "config_sha256": digest(cfg), "frame_sha256": [f["sha256"] for f in unit["frames"]],
        "http_status": 200, "error": None, "response": raw, "parsed": parsed, "transport_seconds": .1})
    try:
        run.reconcile_receipts(phase, ledger, plan, cfg, phase.parent)
        assert ledger.state(unit, cfg["kind"]) == (1, parsed)
        immediate_queue(ledger, cfg["kind"], lambda u: pytest.fail("valid receipt resent")).one(unit)
    finally: ledger.close()


@pytest.mark.parametrize("mutation", ["weights", "frame", "request", "profile", "source"])
def test_corrupt_input_stops_before_backend(frozen, tmp_path, mutation):
    bundle, deployment, _ = frozen
    if mutation == "weights": (deployment / "clip/snapshot/synthetic.safetensors").write_bytes(b"changed")
    elif mutation == "frame": (bundle / "frame-0.jpg").write_bytes(b"changed")
    elif mutation == "request":
        path = bundle / "plans/clip-validation.json"
        value = json.loads(path.read_text()); value["units"][0]["request_sha256"] = "0" * 64
        path.write_text(json.dumps(value)); seal(bundle)
    elif mutation == "profile":
        path = bundle / "configs/clip.json"
        value = json.loads(path.read_text()); value["window_frames"] = 2
        path.write_text(json.dumps(value)); seal(bundle)
    else:
        path = bundle / "source-snapshot/tools/unified_benchmark/native_assets/videojev.patch"
        path.write_text("changed")
    with pytest.raises(ValueError): execute(frozen, tmp_path / "run")
    assert not FakeBackend.instances


def test_calibration_deployment_mismatch_blocks_test(frozen, tmp_path):
    out = tmp_path / "val"
    execute(frozen, out, phase="validation")
    path = out / "thresholds.json"
    artifact = json.loads(path.read_text()); artifact["native_deployment_sha256"] = "0" * 64
    changed = tmp_path / "changed.json"; write_new(changed, artifact)
    FakeBackend.instances.clear()
    with pytest.raises(ValueError, match="deployment differs"):
        execute(frozen, tmp_path / "test", phase="test", threshold_path=changed, threshold_hash=file_hash(changed))
    assert not FakeBackend.instances


def test_separate_validation_then_test_accepts_only_its_frozen_artifact(frozen, tmp_path):
    out = tmp_path / "val"
    execute(frozen, out, phase="validation")
    path = out / "thresholds.json"
    execute(frozen, tmp_path / "test", phase="test", threshold_path=path, threshold_hash=file_hash(path))
    assert json.loads((tmp_path / "test/test/report.json").read_text())["correct"] == 120


def test_startup_failure_does_not_consume_request_budget(frozen, tmp_path, monkeypatch):
    def fail(self): raise RuntimeError("synthetic startup failure")
    monkeypatch.setattr(FakeBackend, "start", fail)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="startup"):
        execute(frozen, out)
    assert (out / "validation/attempts.jsonl").read_bytes() == b""
    assert FakeBackend.calls == [] and all(b.closed for b in FakeBackend.instances)
    monkeypatch.setattr(FakeBackend, "start", lambda self: None)
    execute(frozen, out, resume=True)
    assert len(FakeBackend.calls) == 144


def test_changed_resume_phase_rejected_before_start(frozen, tmp_path):
    out = tmp_path / "run"
    execute(frozen, out, "videojev", "validation")
    FakeBackend.instances.clear()
    with pytest.raises(ValueError, match="resume run identity"):
        execute(frozen, out, "videojev", "both", resume=True)
    assert FakeBackend.instances == []


def test_checkpoint_download_checks_official_bytes_without_network(tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace
    cfg = deploy.profile("clip")
    snapshot = tmp_path / "snapshot"; snapshot.mkdir()
    weight = snapshot / "model.safetensors"; weight.write_bytes(b"synthetic weight")
    config = snapshot / "config.json"; config.write_bytes(b"{}")
    sha = file_hash(weight)
    info = SimpleNamespace(sha=cfg["revision"], siblings=[
        SimpleNamespace(rfilename=weight.name, size=weight.stat().st_size, lfs={"sha256": sha}, blob_id=None),
        SimpleNamespace(rfilename=config.name, size=2, lfs=None, blob_id=hashlib.sha1(b"blob 2\0{}").hexdigest())])
    fake = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *a, **kw: info),
                           snapshot_download=lambda *a, **kw: str(snapshot))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    deploy.download_checkpoint("clip", tmp_path / "cache", tmp_path / "verified.json")
    deploy.verify_checkpoint(json.loads((tmp_path / "verified.json").read_text()))
    weight.write_bytes(b"corrupted weight")
    with pytest.raises(ValueError, match="size differs|checksum differs"):
        deploy.download_checkpoint("clip", tmp_path / "cache", tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


def test_per_video_order_and_twelve_worker_limit():
    lock = threading.Lock()
    active, peak, sources, seen = [0], [0], set(), []
    class FakeQueue:
        stopped = threading.Event()
        def one(self, unit):
            with lock:
                assert unit["source_id"] not in sources
                sources.add(unit["source_id"]); active[0] += 1; peak[0] = max(peak[0], active[0])
            time.sleep(.002)
            with lock:
                seen.append(unit["id"]); sources.remove(unit["source_id"]); active[0] -= 1
    run.video_groups([{"id": f"{s}/{i}", "source_id": str(s)} for s in range(20) for i in range(2)], FakeQueue())
    assert 1 < peak[0] <= 12
    assert all(seen.index(f"{s}/0") < seen.index(f"{s}/1") for s in range(20))


@pytest.mark.parametrize("alias", ["videojev", "decision3"])
def test_reconstruct_exact_historical_source_offline(tmp_path, alias):
    # Public Git objects only; fetch/download is never called by export_service.
    hashes = deploy.export_service(alias, tmp_path / alias)
    assert len(hashes) > 100
    assert all(file_hash(tmp_path / alias / name) == sha for name, sha in hashes.items())


def test_plan_only_deployment_never_starts_installer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(deploy, "prepare", lambda *a: pytest.fail("unapproved install"))
    deploy.main(["prepare", "--root", str(tmp_path / "not-created")])
    assert json.loads(capsys.readouterr().out)["plan_only"]
    assert not (tmp_path / "not-created").exists()


def test_offline_native_cli_no_runtime(frozen, tmp_path, capsys):
    bundle, deployment, sha = frozen
    run.main(["--model", "videojev", "--bundle", str(bundle), "--deployment", str(deployment),
              "--deployment-sha256", sha, "--output", str(tmp_path / "untouched")])
    assert json.loads(capsys.readouterr().out)["inference_calls"] == 0
    assert not (tmp_path / "untouched").exists()
