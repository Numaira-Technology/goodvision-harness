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
    models = {}
    for path in (bundle / "plans").glob("*.json"):
        alias, split = path.stem.rsplit("-", 1)
        models.setdefault(alias, {})[split] = {"plan_sha256": digest(json.loads(path.read_text()))}
    write_new(bundle / "preflight.json", {"offline_passed": True, "sources_sha256": digest(sources), "models": models})
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


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("bad_raw", [None, [], {"answers": None}, {"answers": {"fight": None}}])
def test_malformed_decision3_retries_and_can_resume(frozen, tmp_path, bad_raw, recover):
    target = "test/fight/0/yes"
    def behavior(self, unit, body):
        if unit["id"] == target and (not recover or self.calls.count(target) == 1):
            return 200, self.raw(unit, "yes") | {"raw": bad_raw}, {}
    FakeBackend.behavior = behavior
    out = tmp_path / "run"
    execute(frozen, out, "decision3", "test")
    expected = 2 if recover else 3
    assert Counter(FakeBackend.calls)[target] == expected
    result = json.loads((out / "test/report.json").read_text())
    assert result["correct"] == (120 if recover else 119)
    assert result["final_missing_unique_units"] == (0 if recover else 1)
    execute(frozen, out, "decision3", "test", resume=True)
    assert Counter(FakeBackend.calls)[target] == expected


@pytest.mark.parametrize("mutation", ["delete", "edit", "remove_hash"])
def test_failed_receipt_tampering_blocks_resume_before_start(frozen, tmp_path, mutation):
    target = "test/fight/0/yes"
    FakeBackend.behavior = lambda self, unit, body: (403, {"error": "stop"}, {}) if unit["id"] == target else None
    out = tmp_path / "run"
    with pytest.raises(PausedRun):
        execute(frozen, out, "decision3", "test")
    path = out / "test/receipts" / f"{digest(target)}-1.json"
    ledger_path = out / "test/attempts.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    row = next(r for r in rows if r["unit"] == target and r["type"] == "result")
    assert row["metadata"]["receipt_sha256"] == file_hash(path)
    if mutation == "delete":
        path.unlink()
    elif mutation == "edit":
        path.write_text(path.read_text().replace('"stop"', '"changed"'))
    else:
        row["metadata"].pop("receipt_sha256")
        ledger_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    FakeBackend.instances.clear()
    with pytest.raises(ValueError, match="saved receipt"):
        execute(frozen, out, "decision3", "test", resume=True)
    assert FakeBackend.instances == []


@pytest.mark.parametrize("attempt, retry_after, delay", [(1, 0, 30), (2, 0, 60), (1, 90, 90)])
def test_orphan_failure_restores_receipt_and_remaining_cooldown(frozen, tmp_path, attempt, retry_after, delay):
    bundle, _, _ = frozen
    cfg = deploy.profile("decision3")
    plan = json.loads((bundle / "plans/decision3-test.json").read_text())
    unit = plan["units"][0]
    phase = tmp_path / "run/test"
    ledger = Ledger(phase / "attempts.jsonl", workflow.context_for(plan, cfg, workflow.verify_bundle(bundle)))
    for number in range(1, attempt + 1):
        ledger.append({"unit": unit["id"], "attempt": number, "type": "intent", "request_sha256": unit["request_sha256"]})
    path = phase / "receipts" / f"{digest(unit['id'])}-{attempt}.json"
    write_new(path, {"unit": unit["id"], "attempt": attempt, "request_sha256": unit["request_sha256"],
        "config_sha256": digest(cfg), "frame_sha256": [f["sha256"] for f in unit["frames"]],
        "http_status": 503, "error": "HTTP 503", "response": {"error": "unavailable"}, "parsed": {},
        "transport_seconds": .1, "completed_at_unix": 1000, "fatal": False, "retry_after": retry_after})
    try:
        run.reconcile_receipts(phase, ledger, plan, cfg, phase.parent)
        row = ledger.rows[-1]
        assert row["metadata"]["receipt_sha256"] == file_hash(path)
        assert row["cooldown_until"] == 1000 + delay and row["http_status"] == 503
        tick, sent = [1005.0], []
        def send(_):
            sent.append(tick[0])
            return {"fight": {"answer": "yes"}}, {}
        queue = Queue(ledger, "choice", send, clock=lambda: tick[0], sleep=lambda s: tick.__setitem__(0, tick[0] + s))
        queue.one(unit)
        assert sent == [1000 + delay]
        assert ledger.state(unit, "choice")[0] == attempt + 1
    finally:
        ledger.close()


@pytest.mark.parametrize("failure", ["cuda", "memory_file"])
def test_encoder_always_closed_when_telemetry_fails(tmp_path, monkeypatch, failure):
    import sys
    from types import SimpleNamespace
    backend = run.NativeBackend("clip", {}, tmp_path, tmp_path, tmp_path, 8093)
    closed = []
    backend.encoder = SimpleNamespace(close=lambda: closed.append(True))
    def cuda_query():
        if failure == "cuda":
            raise RuntimeError("synthetic CUDA failure")
        return 0
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        max_memory_allocated=cuda_query, max_memory_reserved=lambda: 0)))
    original_write = run.write_new
    def write(path, value):
        if failure == "memory_file" and path.name == "memory.json":
            raise OSError("synthetic memory write failure")
        original_write(path, value)
    monkeypatch.setattr(run, "write_new", write)
    with pytest.raises((RuntimeError, OSError), match="synthetic"):
        backend.close()
    assert closed == [True] and backend.stop.is_set()


@pytest.mark.parametrize("name", ["plans/decision3-test.json", "configs/decision3.json", "frame-0.jpg"])
def test_unsealed_native_input_rejected_before_backend(frozen, tmp_path, name):
    bundle, _, _ = frozen
    path = bundle / "bundle-files-sha256.json"
    manifest = json.loads(path.read_text())
    manifest.pop(name)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsealed bundle"):
        execute(frozen, tmp_path / "run", "decision3", "test")
    assert not FakeBackend.instances


@pytest.mark.parametrize("alias", deploy.ALIASES)
@pytest.mark.parametrize("huge", [10 ** 400, -(10 ** 400), float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("recover", [False, True])
def test_oversized_native_numbers_use_bounded_retries(frozen, tmp_path, alias, huge, recover):
    target = "test/fight/0/yes"
    def behavior(self, unit, body):
        if unit["id"] != target or (recover and self.calls.count(target) > 1):
            return None
        raw = self.raw(unit, "yes")
        if self.cfg["kind"] == "margin":
            raw["fight"]["margin"] = huge
        elif self.alias == "videojev":
            raw["raw"][0]["probabilities"][0] = huge
        else:
            raw["raw"]["answers"]["fight"]["probabilities"][D3_OPTIONS[0]] = huge
        return 200, raw, {}
    FakeBackend.behavior = behavior
    out = tmp_path / "run"
    execute(frozen, out, alias)
    count = 2 if recover else 3
    assert Counter(FakeBackend.calls)[target] == count
    report = json.loads((out / "test/report.json").read_text())
    assert report["correct"] == (120 if recover else 119)
    assert report["incomplete_questions"] == (0 if recover else 1)
    assert report["accuracy_denominator"] == 120
    assert len(list((out / "test/receipts").glob(f"{digest(target)}-*.json"))) == count
    execute(frozen, out, alias, resume=True)
    assert Counter(FakeBackend.calls)[target] == count


def seeded_native_resume(frozen, output, *, fatal=True, completed_at=1000, recorded=False):
    """Crash boundary: durable intent and receipt, optionally its ledger result."""
    bundle, deployment, deployment_sha = frozen
    cfg = deploy.profile("decision3")
    plan = json.loads((bundle / "plans/decision3-test.json").read_text())
    unit = plan["units"][0]
    context = workflow.context_for(plan, cfg, workflow.verify_bundle(bundle))
    write_new(output / "run.json", {"model": "decision3", "phase": "test",
        "bundle_sha256": workflow.verify_bundle(bundle), "deployment_sha256": deployment_sha,
        "initial_threshold_sha256": None})
    ledger = Ledger(output / "test/attempts.jsonl", context)
    base = {"unit": unit["id"], "request_sha256": unit["request_sha256"], "attempt": 1}
    ledger.append(base | {"type": "intent"})
    receipt = base | {"config_sha256": digest(cfg), "frame_sha256": [f["sha256"] for f in unit["frames"]],
        "response": None, "parsed": {}, "http_status": 401 if fatal else 503,
        "error": "HTTP 401" if fatal else "HTTP 503", "transport_seconds": .01,
        "fatal": fatal, "retry_after": 0}
    if completed_at != "missing":
        receipt["completed_at_unix"] = completed_at
    path = output / "test/receipts" / f"{digest(unit['id'])}-1.json"
    write_new(path, receipt)
    if recorded:
        ledger.append(base | {"type": "result", "answers": {}, "complete": False,
            "elapsed_seconds": .01, "error": receipt["error"], "fatal": fatal,
            "http_status": receipt["http_status"], "metadata": {
                "receipt": str(path.relative_to(output)), "receipt_sha256": file_hash(path)}})
    ledger.close()
    return unit, path


def test_newly_recovered_fatal_pauses_before_backend_then_explicit_resume(frozen, tmp_path):
    output = tmp_path / "recover-fatal"
    unit, path = seeded_native_resume(frozen, output, completed_at="missing")
    receipt_hash = file_hash(path)
    intent_bytes = (output / "test/attempts.jsonl").read_bytes()
    with pytest.raises(PausedRun, match="newly recovered fatal receipt"):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances and not FakeBackend.calls
    rows = [json.loads(line) for line in (output / "test/attempts.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[-1]["fatal"] is True
    assert rows[-1]["metadata"]["receipt_sha256"] == receipt_hash == file_hash(path)
    assert rows[-1]["metadata"]["recovered_receipt"] is True
    assert (output / "test/attempts.jsonl").read_bytes().startswith(intent_bytes)
    assert list(output.glob("invocations/*/test-partial-report.json"))
    assert not (output / "test/completion.json").exists()
    # A subsequent explicit resume acknowledges the now recorded stop, not a
    # permanent prohibition or a fresh three-attempt budget.
    execute(frozen, output, "decision3", "test", resume=True)
    assert len(FakeBackend.calls) == 120 and Counter(FakeBackend.calls)[unit["id"]] == 1
    rows = [json.loads(line) for line in (output / "test/attempts.jsonl").read_text().splitlines()]
    assert [r["attempt"] for r in rows if r["type"] == "intent" and r["unit"] == unit["id"]] == [1, 2]
    assert file_hash(path) == receipt_hash
    assert json.loads((output / "test/report.json").read_text())["correct"] == 120


def test_already_recorded_fatal_allows_explicit_resume(frozen, tmp_path):
    output = tmp_path / "recorded-fatal"
    unit, path = seeded_native_resume(frozen, output, recorded=True)
    before = (output / "test/attempts.jsonl").read_bytes()
    receipt_hash = file_hash(path)
    execute(frozen, output, "decision3", "test", resume=True)
    assert len(FakeBackend.calls) == 120
    assert (output / "test/attempts.jsonl").read_bytes().startswith(before)
    assert file_hash(path) == receipt_hash
    assert json.loads((output / "test/report.json").read_text())["correct"] == 120


@pytest.mark.parametrize("bad_time", ["missing", None, False, "1000", float("inf"), float("nan"), 10 ** 400],
                         ids=["missing", "null", "boolean", "string", "infinite", "nan", "oversized"])
def test_incomplete_orphan_requires_reliable_completion_time_before_backend(frozen, tmp_path, bad_time):
    output = tmp_path / "legacy-incomplete"
    _, path = seeded_native_resume(frozen, output, fatal=False)
    receipt = json.loads(path.read_text())
    if bad_time == "missing":
        receipt.pop("completed_at_unix")
    else:
        receipt["completed_at_unix"] = bad_time
    # Legacy/corrupt external receipt, including noncanonical numeric tokens.
    path.write_text(json.dumps(receipt))
    before = (output / "test/attempts.jsonl").read_bytes()
    receipt_hash = file_hash(path)
    with pytest.raises(ValueError, match="saved incomplete receipt requires finite completed_at_unix"):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances and not FakeBackend.calls
    assert (output / "test/attempts.jsonl").read_bytes() == before
    assert file_hash(path) == receipt_hash
    assert not (output / "test/completion.json").exists()


@pytest.mark.parametrize("corruption", ["unmatched", "result_first", "duplicate"])
def test_invalid_result_identity_stops_actual_native_resume_before_backend(frozen, tmp_path, corruption):
    output = tmp_path / "invalid-result"
    unit, path = seeded_native_resume(frozen, output, recorded=True)
    ledger_path = output / "test/attempts.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    if corruption == "unmatched":
        rows[-1]["attempt"] = 2
        rows[-1]["answers"] = {"fight": {"answer": "yes"}}
    elif corruption == "result_first":
        rows.reverse()
    else:
        rows.append(rows[-1])
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = ledger_path.read_bytes()
    with pytest.raises(ValueError, match="result.*preceding intent|duplicate result"):
        execute(frozen, output, "decision3", "test", resume=True)
    assert not FakeBackend.instances and not FakeBackend.calls
    assert ledger_path.read_bytes() == before
    assert not (output / "test/completion.json").exists()


def test_late_orphan_keeps_first_attempt_answer_in_actual_native_resume(frozen, tmp_path):
    from tools.unified_benchmark.adapters import parse_response
    bundle, deployment, _ = frozen
    output = tmp_path / "late-orphan"
    unit, first_path = seeded_native_resume(frozen, output, fatal=False)
    cfg = deploy.profile("decision3")
    plan = json.loads((bundle / "plans/decision3-test.json").read_text())
    context = workflow.context_for(plan, cfg, workflow.verify_bundle(bundle))
    backend = FakeBackend("decision3", cfg, bundle, deployment, output, 1)
    first_raw, second_raw = backend.raw(unit, "yes"), backend.raw(unit, "no")
    first, second = parse_response(cfg, unit, first_raw), parse_response(cfg, unit, second_raw)
    receipt = json.loads(first_path.read_text()) | {"http_status": 200, "error": None,
                                                   "response": first_raw, "parsed": first}
    first_path.write_text(json.dumps(receipt))
    second_path = first_path.with_name(f"{digest(unit['id'])}-2.json")
    write_new(second_path, receipt | {"attempt": 2, "response": second_raw, "parsed": second})
    ledger = Ledger(output / "test/attempts.jsonl", context)
    base = {"unit": unit["id"], "request_sha256": unit["request_sha256"], "attempt": 2}
    ledger.append(base | {"type": "intent"})
    ledger.append(base | {"type": "result", "answers": second, "complete": True,
        "elapsed_seconds": .01, "metadata": {"receipt": str(second_path.relative_to(output)),
                                              "receipt_sha256": file_hash(second_path)}})
    ledger.close()
    before = (output / "test/attempts.jsonl").read_bytes()
    hashes = {path: file_hash(path) for path in (first_path, second_path)}
    FakeBackend.instances.clear()
    execute(frozen, output, "decision3", "test", resume=True)
    assert len(FakeBackend.calls) == 119 and unit["id"] not in FakeBackend.calls
    ledger = Ledger(output / "test/attempts.jsonl", context)
    try:
        assert ledger.state(unit, cfg["kind"]) == (2, first)
        assert [row["attempt"] for row in ledger.rows
                if row["type"] == "result" and row["unit"] == unit["id"]] == [2, 1]
    finally:
        ledger.close()
    assert (output / "test/attempts.jsonl").read_bytes().startswith(before)
    assert all(file_hash(path) == sha for path, sha in hashes.items())
    assert json.loads((output / "test/report.json").read_text())["correct"] == 120
    completed = (output / "test/attempts.jsonl").read_bytes()
    execute(frozen, output, "decision3", "test", resume=True)
    assert len(FakeBackend.calls) == 119
    assert (output / "test/attempts.jsonl").read_bytes() == completed
