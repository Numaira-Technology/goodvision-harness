"""Portable execution of the recorded four-model native 120-question protocol.

Twelve video workers; exactly one in-flight window per source. No paced drops,
new prompt policy, single-frame conversion, or changed model precision.
"""
import argparse
import base64
import fcntl
import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .adapters import frame_bytes, parse_response, request_body
from .contract import (
    EVENTS,
    RULES,
    canonical,
    digest,
    file_hash,
    finite_number,
    select_thresholds,
    write_new,
)
from .native_deploy import ALIASES, environment_for, profile, verify_deployment
from .receipts import raw_fields, read_receipt
from .retry import Ledger, PausedRun, Queue, RequestFailure
from .workflow import (
    collect,
    context_for,
    load_threshold_artifact,
    report,
    verify_bundle,
)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_status(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def video_groups(units, queue, after=lambda: None):
    groups = defaultdict(list)
    if len({u["id"] for u in units}) != len(units):
        raise ValueError("duplicate planned units")
    for unit in units:
        groups[unit["source_id"]].append(unit)
    def run(group):
        for unit in group:
            queue.one(unit)
            after()
    with ThreadPoolExecutor(max_workers=12) as pool:
        try:
            list(pool.map(run, groups.values()))
        except BaseException:
            queue.stopped.set()
            raise


def service_command(alias, deployment, output, port):
    app = deployment / alias / "app"
    py = str(deployment / alias / "env/bin/python")
    env = dict(os.environ, PYTHONPATH=str(app / "src"), PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_XET="1",
               PYTHONDONTWRITEBYTECODE="1")
    if alias == "videojev":
        args = [py, str(app / "private-eval/service.py"), "--ready-capacity", "4",
                "--gpu-batch-size", "4", "--max-pixels", "409600", "--prefix-cache"]
    else:
        args = [py, "-m", "goodvision_harness.videojev_service", "--runtime", "decision3",
                "--ready-capacity", "2", "--gpu-batch-size", "2", "--max-pixels", "200704",
                "--no-prefix-cache", "--no-d3-shared-window-state", "--host", "127.0.0.1", "--cpu-threads", "4"]
    args += ["--cache-dir", str(deployment / alias / "cache"), "--result-log", str(output / "service-results.jsonl"),
             "--port", str(port), "--request-timeout", "20", "--queue-age", "19.5",
             "--queue-capacity", "48", "--context", "32768"]
    return args, env


def validate_health(alias, config, health):
    runtime = health.get("runtime", {})
    if health.get("ready") is not True:
        raise ValueError("native service is not ready")
    for key in ("model", "revision"):
        if runtime.get(key) != config[key] or health.get(key) != config[key]:
            raise ValueError("native service identity differs")
    if ("4090" not in runtime.get("gpu", "")
            or not 23 * 1024**3 <= runtime.get("gpu_total_bytes", 0) <= 25 * 1024**3
            or runtime.get("cuda_runtime") != "12.8"):
        raise ValueError("RTX 4090 24GB / CUDA12.8 required")
    expected = {"gpu_batch_size": 4 if alias == "videojev" else 2,
                "max_pixels": 409600 if alias == "videojev" else 200704,
                "queue_policy": "latest-per-stream"}
    if any(health.get(k) != v for k, v in expected.items()):
        raise ValueError("native service configuration differs")
    versions = runtime.get("versions", {})
    if (versions.get("torch") != "2.8.0+cu128"
            or versions.get("transformers") != ("5.14.1" if alias == "videojev" else "5.17.0")
            or runtime.get("prefix_cache") is not (alias == "videojev")
            or runtime.get("torch_cpu_threads") != 4):
        raise ValueError("native runtime environment differs")
    if alias == "decision3" and (runtime.get("shared_window_state") is not False
            or runtime.get("precision") != "BF16 backbone, FP32 readout"):
        raise ValueError("Decision3 execution mode differs")
    if alias == "videojev" and (runtime.get("attention") != "flash_attention_2" or runtime.get("context") != 32768):
        raise ValueError("VideoJev attention/context differs")


class NativeBackend:
    def __init__(self, alias, config, bundle, deployment, output, port):
        self.alias, self.config, self.bundle = alias, config, bundle
        self.deployment, self.output, self.port = deployment, output, port
        self.encoder = self.service = self.log = None
        self.gpu_lock = threading.Lock()
        self.stop = threading.Event()
        self.monitor = None
        self.peak_mib = None

    def health(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=3) as r:
            return json.load(r)

    def http(self, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/predict", data=canonical(body),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, {"error": e.read().decode(errors="replace")}

    def telemetry(self):
        with (self.output / "gpu-telemetry.jsonl").open("x") as f:
            while not self.stop.is_set():
                row = {"at": stamp()}
                try:
                    p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                    memory, usage, temperature = [int(s.strip()) for s in p.stdout.strip().split(",")]
                    self.peak_mib = max(self.peak_mib or 0, memory)
                    row.update(memory_mib=memory, utilization=usage, temperature=temperature)
                except (ValueError, OSError, subprocess.TimeoutExpired):
                    row["error"] = "GPU telemetry unavailable"
                f.write(json.dumps(row) + "\n")
                f.flush()
                self.stop.wait(2)

    def start(self):
        from PIL import Image
        env_dir = self.deployment / environment_for(self.alias) / "env"
        packages = json.loads(subprocess.check_output([str(env_dir / "bin/python"), "-m", "pip", "list", "--format=json"], text=True))
        recorded = json.loads((env_dir.parent / "packages.json").read_text())
        if {p["name"].lower(): p["version"] for p in packages} != {p["name"].lower(): p["version"] for p in recorded}:
            raise ValueError("installed packages differ from frozen deployment")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader"], text=True)
        if len(gpu.strip().splitlines()) != 1 or "4090" not in gpu:
            raise ValueError("one RTX4090 required")
        write_new(self.output / "hardware.json", {"gpu": gpu, "at": stamp(), "host": os.uname().nodename})
        self.monitor = threading.Thread(target=self.telemetry, daemon=True)
        self.monitor.start()
        if self.alias in ("clip", "siglip2"):
            if Path(sys.prefix).resolve() != env_dir.resolve():
                raise ValueError("run contrastive evaluation using the prepared clip/env/bin/python")
            import torch

            from tools.clip_benchmark.runtime import ClipEncoder
            props = torch.cuda.get_device_properties(0)
            if not 23 * 1024**3 <= props.total_memory <= 25 * 1024**3:
                raise ValueError("24GB GPU required")
            torch.set_num_threads(1)
            self.encoder = ClipEncoder(self.config["model"], env_dir.parent / "cache")
            metadata = self.encoder.metadata
            if (metadata["revision"] != self.config["revision"] or metadata["prompts"] != self.config["prompts"]
                    or metadata["precision"] != "float32" or metadata["tf32"] is not False):
                raise ValueError("contrastive encoder configuration differs")
            torch.cuda.reset_peak_memory_stats()
            raw = self.encoder.predict(self.encoder.prepare(Image.new("RGB", (720, 406), (127, 127, 127))))
            parsed = parse_response(self.config, {"targets": list(EVENTS)}, raw)
        else:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", self.port))
            args, env = service_command(self.alias, self.deployment, self.output, self.port)
            self.log = (self.output / "service.log").open("xb")
            self.service = subprocess.Popen(args, env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 1200
            while True:
                if self.service.poll() is not None:
                    raise RuntimeError("native service exited during original warmup; see service.log")
                if time.monotonic() >= deadline:
                    raise TimeoutError("native service startup deadline exceeded")
                try:
                    metadata = self.health()
                    if metadata.get("ready"):
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            validate_health(self.alias, self.config, metadata)
            jpeg = io.BytesIO()
            Image.new("RGB", (720, 406), (127, 127, 127)).save(jpeg, format="JPEG")
            uid = self.alias + "-unified-synthetic-smoke"
            body = {"session": uid, "request_id": uid, "fps": 1.0, "frame_times": [0.0, 1.0],
                    "images": [base64.b64encode(jpeg.getvalue()).decode()] * 2,
                    "question_template": self.config["question_template"]}
            code, raw = self.http(body)
            if code != 200:
                raise RuntimeError("native synthetic smoke failed")
            parsed = parse_response(self.config, {"id": uid, "targets": list(EVENTS)}, raw)
        if set(parsed) != set(EVENTS):
            raise ValueError("synthetic smoke did not produce six valid choices/scores")
        write_new(self.output / "runtime.json", metadata)
        write_new(self.output / "smoke.json", {"passed": True, "synthetic": True, "raw": raw, "parsed": parsed})

    def infer(self, unit, body):
        if self.encoder:
            from PIL import Image
            started = time.monotonic()
            with Image.open(io.BytesIO(frame_bytes(self.bundle, unit["frames"][0]))) as image:
                prepared = self.encoder.prepare(image.convert("RGB"))
            prepared_at = time.monotonic()
            with self.gpu_lock:
                acquired = time.monotonic()
                raw = self.encoder.predict(prepared)
            return 200, raw, {"prepare_seconds": prepared_at - started, "gpu_queue_seconds": acquired - prepared_at,
                              "gpu_readback_seconds": time.monotonic() - acquired}
        code, raw = self.http(body)
        return code, raw, {}

    def close(self):
        try:
            if self.encoder:
                try:
                    import torch
                    write_new(self.output / "memory.json", {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
                finally:
                    self.encoder.close()
            if self.service and self.service.poll() is None:
                try:
                    write_new(self.output / "final-health.json", self.health())
                except Exception:
                    pass
                os.killpg(self.service.pid, signal.SIGINT)
                try:
                    self.service.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(self.service.pid, signal.SIGTERM)
                    try:
                        self.service.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.service.pid, signal.SIGKILL)
                        self.service.wait()
        finally:
            if self.log:
                self.log.close()
            self.stop.set()
            if self.monitor:
                self.monitor.join(5)
            write_new(self.output / "gpu-peak.json", {"max_gpu_used_mib": self.peak_mib, "sampling_seconds": 2})


def validate_inputs(bundle, alias, phase, deployment_hash, threshold_path=None, threshold_hash=None):
    bundle_digest = verify_bundle(bundle)
    cfg = json.loads((bundle / "configs" / (alias + ".json")).read_text())
    if cfg != profile(alias):
        raise ValueError("native profile differs from the recorded protocol")
    phases = ("validation", "test") if phase == "both" else (phase,)
    for split in phases:
        plan = json.loads((bundle / "plans" / f"{alias}-{split}.json").read_text())
        for unit in plan["units"]:
            if digest(request_body(cfg, unit, bundle)[1]) != unit["request_sha256"]:
                raise ValueError("frozen native request differs")
    artifact = None
    if phase == "test" and cfg["kind"] == "margin":
        artifact = load_threshold_artifact(threshold_path, threshold_hash, bundle=bundle,
            alias=alias, config=cfg, bundle_digest=bundle_digest)
        if artifact.get("native_deployment_sha256") != deployment_hash:
            raise ValueError("threshold native deployment differs")
    return cfg, bundle_digest, artifact


def _verify_sealed_receipts(output):
    """Check immutable receipt bytes before a resume can replace the result seal.

    Ledgers/status can grow during an interrupted resume; receipts are create-once.
    An absent seal is allowed for first-invocation crash recovery, but a recorded
    failed result then needs its own receipt hash below.
    """
    manifest = output / "result-files-sha256.json"
    if not manifest.exists():
        return {}
    hashes = json.loads(manifest.read_text())
    receipts = {}
    for name, expected in hashes.items():
        relative = Path(name)
        if "receipts" not in relative.parts:
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid sealed receipt path")
        path = output / relative
        if (
            not path.resolve().is_relative_to(output.resolve())
            or not path.is_file()
            or path.is_symlink()
            or file_hash(path) != expected
        ):
            raise ValueError("saved receipt differs from retained seal: " + name)
        receipts[name] = expected
    return receipts


def _recovery_outcome(receipt, path, *, complete, require_metadata=True):
    """Use authoritative outcome facts; never invent legacy completion times."""
    status, error = receipt.get("http_status"), receipt.get("error")
    if status is not None and (type(status) is not int or not 100 <= status <= 599):
        raise ValueError(f"invalid receipt http_status: {path}")
    if error is not None and not isinstance(error, str):
        raise ValueError(f"invalid receipt error: {path}")
    if (
        not finite_number(receipt.get("transport_seconds"))
        or receipt["transport_seconds"] < 0
    ):
        raise ValueError(f"invalid receipt transport_seconds: {path}")
    required = ("fatal", "retry_after", "completed_at_unix")
    if not complete and require_metadata:
        missing = [key for key in required if key not in receipt]
        if missing:
            raise ValueError(
                f"saved incomplete receipt requires finite completed_at_unix and authoritative outcome metadata; cannot recover incomplete receipt {path}: missing authoritative "
                + ", ".join(missing)
            )
    fatal = receipt.get("fatal", False)
    if type(fatal) is not bool:
        raise ValueError(f"invalid receipt fatal: {path}")
    if fatal and complete:
        raise ValueError(f"fatal receipt cannot contain a complete answer: {path}")
    for field in ("retry_after", "completed_at_unix"):
        if field in receipt and (
            not finite_number(receipt[field]) or receipt[field] < 0
        ):
            raise ValueError(f"saved incomplete receipt requires finite completed_at_unix; invalid receipt {field}: {path}")
    if status in (401, 403) and not fatal:
        raise ValueError(f"authentication failure receipt is not marked fatal: {path}")
    return {
        "http_status": status,
        "fatal": fatal,
        "retry_after": receipt.get("retry_after", 0),
    }


def reconcile_receipts(phase_dir, ledger, plan, cfg, output, sealed_receipts=None, *, recover=True):
    """Verify old answers and recover orphans; newly observed fatal pauses first.

    A fatal already recorded before an explicit resume is acknowledged by that
    invocation. A newly recovered fatal is appended, then surfaced to the caller
    before any backend startup. A subsequent explicit resume may acknowledge it.
    """
    units = {u["id"]: u for u in plan["units"]}
    collect(plan, cfg, ledger)
    results = {
        (r["unit"], r["attempt"]): r for r in ledger.rows if r["type"] == "result"
    }
    if (phase_dir / "completion.json").exists():
        recorded = {phase_dir / "receipts" / f"{digest(unit)}-{attempt}.json"
                    for unit, attempt in results}
        if any(p.is_file() and p not in recorded for p in (phase_dir / "receipts").rglob("*")):
            raise ValueError("completed phase has unrecorded receipts")
    recovered_fatal = []
    pending = []
    for row in list(ledger.rows):
        if row["type"] != "intent":
            continue
        unit = units[row["unit"]]
        path = phase_dir / "receipts" / f"{digest(unit['id'])}-{row['attempt']}.json"
        validate_saved_path(path, output)
        result = results.get((row["unit"], row["attempt"]))
        name = str(path.relative_to(output))
        if result:
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict) or metadata.get("receipt") not in (
                None,
                name,
            ):
                raise ValueError("saved receipt path differs from its attempt")
            # Only historical rows without metadata may use the retained seal.
            # A partially stripped modern anchor is evidence corruption.
            expected_hash = metadata.get("receipt_sha256")
            if "metadata" not in result:
                expected_hash = (sealed_receipts or {}).get(name)
            if (
                not expected_hash
                or not path.is_file()
                or file_hash(path) != expected_hash
            ):
                raise ValueError("saved receipt differs or lacks an integrity anchor")
        elif not path.exists():
            continue  # interrupted send still consumes one attempt
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            raise ValueError("invalid saved receipt path")
        try:
            receipt = read_receipt(path)
        except ValueError as error:
            raise ValueError(
                "saved incomplete receipt requires finite completed_at_unix and strict raw evidence: " + str(error)
            ) from error
        expected = {
            "unit": unit["id"],
            "attempt": row["attempt"],
            "request_sha256": unit["request_sha256"],
            "config_sha256": digest(cfg),
            "frame_sha256": [f["sha256"] for f in unit["frames"]],
        }
        if type(receipt.get("attempt")) is not int or any(
            receipt.get(k) != v for k, v in expected.items()
        ):
            raise ValueError("saved receipt identity differs")
        required = ("http_status", "error", "response", "parsed", "transport_seconds")
        missing = [key for key in required if key not in receipt]
        if missing:
            raise ValueError(
                f"cannot verify receipt {path}: missing authoritative "
                + ", ".join(missing)
            )
        parsed = (
            parse_response(cfg, unit, receipt["response"])
            if receipt["http_status"] == 200 and not receipt["error"]
            else {}
        )
        if parsed != receipt["parsed"]:
            raise ValueError("saved receipt parsing differs")
        if result:
            if result.get("answers") != parsed:
                raise ValueError("saved receipt and ledger answers differ")
            # Legacy recorded failures may carry authoritative outcome facts in
            # their anchored ledger row rather than the older receipt schema.
            # Validate supplied fields without inventing missing completion time
            # or replacing its already-recorded cooldown.
            _recovery_outcome(
                {
                    **{
                        k: result[k]
                        for k in ("fatal", "retry_after", "http_status")
                        if k in result
                    },
                    **receipt,
                },
                path,
                complete=result.get("complete", False),
                require_metadata=False,
            )
            for key in ("fatal", "http_status", "retry_after"):
                if key in receipt and key in result and receipt[key] != result[key]:
                    raise ValueError("saved receipt and ledger outcome differ: " + key)
            continue
        # Later recorded answers must not make an earlier incomplete receipt
        # appear complete or erase its missing recovery metadata/cooldown.
        if (phase_dir / "completion.json").exists():
            raise ValueError("completed phase contains an unrecorded receipt")
        complete = set(parsed) == set(unit["targets"])
        outcome = _recovery_outcome(
            receipt, path, complete=set(parsed) == set(unit["targets"])
        )
        recovered = {
            "type": "result",
            "unit": unit["id"],
            "attempt": row["attempt"],
            "request_sha256": unit["request_sha256"],
            "answers": parsed,
            "complete": complete and not outcome["fatal"],
            "error": receipt["error"],
            "elapsed_seconds": receipt["transport_seconds"],
            **outcome,
            "metadata": {
                "actual_provider": "self-hosted",
                "receipt": name,
                "receipt_sha256": file_hash(path),
                "recovered_receipt": True,
            },
        }
        if (
            not complete
            and not outcome["fatal"]
            and (row["attempt"] < 3 or outcome["http_status"] == 429)
        ):
            recovered["cooldown_until"] = receipt["completed_at_unix"] + max(
                outcome["retry_after"], 30 if row["attempt"] == 1 else 60
            )
            if not finite_number(recovered["cooldown_until"]):
                raise ValueError(f"invalid receipt cooldown sum: {path}")
        canonical(recovered)
        pending.append(recovered)
        if not recover:
            continue
        ledger.append(recovered)
        if outcome["fatal"]:
            recovered_fatal.append(name)
    if recovered_fatal:
        raise PausedRun(
            "newly recovered fatal receipt; inspect before explicit resume: "
            + ", ".join(recovered_fatal)
        )
    return pending



def _verify_completion(output, phase_dir, *, calibration=False):
    path = phase_dir / "completion.json"
    if not path.exists():
        return
    done = json.loads(path.read_text())
    files = done.get("files") if isinstance(done, dict) else None
    required = {
        str((phase_dir / name).relative_to(output))
        for name in ("attempts.jsonl", "report.json", "run.json")
    }
    if calibration:
        required.add("thresholds.json")
    if (
        not isinstance(files, dict)
        or not required <= set(files)
        or done.get("execution_finished") is not True
    ):
        raise ValueError("completed phase is missing evidence anchors")
    for name, expected in files.items():
        relative = Path(name)
        artifact = output / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or str(relative) != name
            or artifact.is_symlink()
            or not artifact.resolve().is_relative_to(output.resolve())
            or not artifact.is_file()
            or file_hash(artifact) != expected
        ):
            raise ValueError("completed phase evidence differs: " + name)


def _verify_existing_phase_records(
    bundle, output, splits, cfg, bundle_digest, deployment_hash, threshold_hash
):
    """Check all phase identities/anchors before recovery can append any result."""
    test_threshold_hash = threshold_hash
    if (
        cfg["kind"] == "margin"
        and "validation" in splits
        and "test" in splits
        and (output / "test/run.json").exists()
    ):
        saved = output / "thresholds.json"
        if not saved.is_file():
            raise ValueError("test phase requires saved validation thresholds")
        test_threshold_hash = file_hash(saved)
        artifact = load_threshold_artifact(
            saved,
            test_threshold_hash,
            bundle=bundle,
            alias=cfg["alias"],
            config=cfg,
            bundle_digest=bundle_digest,
        )
        if artifact.get("native_deployment_sha256") != deployment_hash:
            raise ValueError("threshold native deployment differs")
    for split in splits:
        phase_dir = output / split
        record_path = phase_dir / "run.json"
        if record_path.exists():
            plan = json.loads(
                (bundle / "plans" / f"{cfg['alias']}-{split}.json").read_text()
            )
            expected = {
                "context": context_for(plan, cfg, bundle_digest),
                "threshold_artifact_sha256": test_threshold_hash
                if split == "test"
                else None,
                "native_deployment_sha256": deployment_hash,
            }
            recorded = json.loads(record_path.read_text())
            if not isinstance(recorded, dict) or any(
                recorded.get(k) != v for k, v in expected.items()
            ):
                raise ValueError("resume phase identity differs")
        _verify_completion(
            output,
            phase_dir,
            calibration=split == "validation" and cfg["kind"] == "margin",
        )


def _reconcile_existing_phases(bundle, output, splits, cfg, bundle_digest, sealed, invocation=None):
    """Validate/recover every requested existing phase before starting a backend."""
    for split in splits:
        phase_dir = output / split
        ledger_path = phase_dir / "attempts.jsonl"
        if not ledger_path.exists():
            continue
        plan = json.loads(
            (bundle / "plans" / f"{cfg['alias']}-{split}.json").read_text()
        )
        ledger = Ledger(ledger_path, context_for(plan, cfg, bundle_digest))
        try:
            reconcile_receipts(phase_dir, ledger, plan, cfg, output, sealed)
        except PausedRun:
            if invocation is not None:
                thresholds = {e: 0.0 for e in EVENTS}
                if (output / "thresholds.json").is_file():
                    thresholds = json.loads((output / "thresholds.json").read_text())["thresholds"]
                write_new(invocation / (split + "-partial-report.json"), report(plan, cfg, ledger, thresholds))
            raise
        finally:
            ledger.close()


def execute(alias, bundle, deployment, output, deployment_hash, *, phase, threshold_path=None,
            threshold_hash=None, port=8093, authorization, resume=False,
            backend_factory=NativeBackend, queue_factory=Queue):
    cfg, bundle_digest, artifact = validate_inputs(bundle, alias, phase, deployment_hash, threshold_path, threshold_hash)
    verify_deployment(deployment, deployment_hash, alias)
    identity = {"model": alias, "phase": phase, "bundle_sha256": bundle_digest,
                "deployment_sha256": deployment_hash, "initial_threshold_sha256": threshold_hash}
    sealed_receipts = _verify_sealed_receipts(output) if resume else {}
    if resume:
        recorded = json.loads((output / "run.json").read_text())
        if not isinstance(recorded, dict) or any(recorded.get(k) != v for k, v in identity.items()):
            raise ValueError("resume run identity differs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_new(output / "run.json", {**identity, "started_at": stamp(), "authorization": authorization, "slots": 12,
            "scheduler": "12 video workers; one in-flight unit per source; ordered windows; backpressure, no planned drops",
            "ground_truth_in_model_requests": False, "not_12_camera_throughput": True})
    splits = ("validation", "test") if phase == "both" else (phase,)
    if resume:
        preflight_resume(alias, bundle, output, cfg, bundle_digest, deployment_hash,
                         phase, threshold_hash, sealed_receipts)
    invocation = output / "invocations" / uuid.uuid4().hex
    invocation.mkdir(parents=True)
    write_new(invocation / "run.json", {"started_at": stamp(), "resume": resume, "authorization": authorization})
    backend = None
    def ensure_backend():
        nonlocal backend
        # Called from worker threads. Start/warm up exactly once per invocation.
        with startup_lock:
            if backend is None:
                backend = backend_factory(alias, cfg, bundle, deployment, invocation, port)
                try:
                    backend.start()
                except BaseException:
                    startup_failed.set()
                    raise
            if startup_failed.is_set():
                raise RequestFailure("native startup failed; see invocation logs", fatal=True)
    startup_lock, startup_failed = threading.Lock(), threading.Event()
    completed = []
    try:
        save_status(output / "status.json", {"state": "starting", "pid": os.getpid(), "at": stamp(), "invocation": str(invocation)})
        if resume:
            _reconcile_existing_phases(
                bundle, output, splits, cfg, bundle_digest, sealed_receipts, invocation
            )
        for split in splits:
            plan = json.loads((bundle / "plans" / f"{alias}-{split}.json").read_text())
            phase_dir = output / split
            phase_dir.mkdir(exist_ok=True)
            thresholds = artifact["thresholds"] if artifact else {e: 0.0 for e in EVENTS}
            if split == "test" and cfg["kind"] == "margin" and artifact is None:
                raise ValueError("validation calibration missing; test blocked")
            phase_identity = {"context": context_for(plan, cfg, bundle_digest),
                "threshold_artifact_sha256": threshold_hash if split == "test" else None,
                "native_deployment_sha256": deployment_hash}
            phase_record = phase_dir / "run.json"
            if phase_record.exists():
                existing = json.loads(phase_record.read_text())
                if any(existing.get(k) != v for k, v in phase_identity.items()):
                    raise ValueError("resume phase identity differs")
            else:
                write_new(phase_record, {**phase_identity, "started_at": stamp()})
            ledger = Ledger(phase_dir / "attempts.jsonl", context_for(plan, cfg, bundle_digest))
            def send(unit):
                body = request_body(cfg, unit, bundle)[1]
                if digest(body) != unit["request_sha256"]:
                    raise RequestFailure("request fingerprint differs", fatal=True)
                attempt = ledger.state(unit, cfg["kind"])[0]
                path = phase_dir / "receipts" / f"{digest(unit['id'])}-{attempt}.json"
                started = time.monotonic()
                raw = None
                code = None
                error = None
                timing = {}
                parsed = {}
                fatal = False
                try:
                    code, raw, timing = backend.infer(unit, body)
                    if code == 200:
                        parsed = parse_response(cfg, unit, raw)
                    else:
                        error = "HTTP " + str(code)
                        fatal = code in (401, 403)
                except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, KeyError, TypeError) as e:
                    error = type(e).__name__ + ": " + str(e)
                    fatal = "identity differs" in error
                except Exception as e:
                    error, fatal = type(e).__name__ + ": " + str(e), True
                write_new(path, {"unit": unit["id"], "attempt": attempt, "request_sha256": unit["request_sha256"],
                    "config_sha256": digest(cfg), "frame_sha256": [f["sha256"] for f in unit["frames"]],
                    "at": stamp(), "http_status": code,
                    "parsed": parsed, "error": error,
                    "fatal": fatal, "retry_after": 0, "completed_at_unix": time.time(),
                    "transport_seconds": time.monotonic() - started,
                    **raw_fields(response=raw, timings=timing)})
                metadata = {
                    "actual_provider": "self-hosted",
                    "receipt": str(path.relative_to(output)),
                    "receipt_sha256": file_hash(path), "http_status": code,
                }
                if error:
                    raise RequestFailure(
                        error, status=code, fatal=fatal, metadata=metadata
                    )
                return parsed, metadata
            try:
                reconcile_receipts(
                    phase_dir, ledger, plan, cfg, output, sealed_receipts
                )
                completion_path = phase_dir / "completion.json"
                if completion_path.exists():
                    _verify_completion(
                        output, phase_dir,
                        calibration=split == "validation" and cfg["kind"] == "margin",
                    )
                else:
                    if any(used < 3 and set(answers) != set(unit["targets"])
                           for unit in plan["units"]
                           for used, answers in [ledger.state(unit, cfg["kind"]) ]):
                        ensure_backend()
                    queue = queue_factory(ledger, cfg["kind"], send, slots=12)
                    save_status(output / "status.json", {"state": "running", "split": split, "pid": os.getpid(), "at": stamp()})
                    video_groups(plan["units"], queue)
                if split == "validation" and cfg["kind"] == "margin":
                    selected = select_thresholds(plan, collect(plan, cfg, ledger), cfg["kind"], cfg["model"])
                    selected.update(calibration="selected_on_complete_validation24", config_sha256=digest(cfg),
                        rules_sha256=digest(RULES), bundle_sha256=bundle_digest, model=cfg["model"],
                        native_deployment_sha256=deployment_hash)
                    threshold_path = output / "thresholds.json"
                    if threshold_path.exists():
                        artifact = json.loads(threshold_path.read_text())
                        if any(digest(artifact.get(k)) != digest(v) for k, v in selected.items()):
                            raise ValueError("saved calibration differs from validation ledger")
                    else:
                        write_new(threshold_path, {**selected, "selected_at": stamp()})
                    threshold_hash = file_hash(threshold_path)
                    artifact = load_threshold_artifact(threshold_path, threshold_hash, bundle=bundle, alias=alias,
                                                      config=cfg, bundle_digest=bundle_digest)
                if not completion_path.exists():
                    result = report(plan, cfg, ledger, thresholds)
                    result["native_deployment_sha256"] = deployment_hash
                    result["threshold_artifact_sha256"] = threshold_hash if split == "test" else None
                    # Incomplete validation still produces a report in the exception handler below.
                    save_status(phase_dir / "report.json", result)
                    files = [ledger.path, phase_dir / "report.json", phase_record]
                    if split == "validation" and cfg["kind"] == "margin":
                        files.append(threshold_path)
                    write_new(completion_path, {"execution_finished": True,
                        "final_missing_unique_units": result["final_missing_unique_units"],
                        "incomplete_questions": result["incomplete_questions"],
                        "files": {str(p.relative_to(output)): file_hash(p) for p in files}, "at": stamp()})
                completed.append(split)
            except BaseException:
                # Per-invocation partial report: never overwrite completed evidence.
                write_new(invocation / (split + "-partial-report.json"), report(plan, cfg, ledger, thresholds))
                raise
            finally:
                ledger.close()
        save_status(output / "status.json", {"state": "completed", "phases": completed, "at": stamp(),
                                             "note": "execution finished; inspect coverage separately"})
    except BaseException as e:
        write_new(invocation / "failure.json", {"error": type(e).__name__ + ": " + str(e), "at": stamp()})
        save_status(output / "status.json", {"state": "failed", "phases": completed, "error": str(e), "at": stamp()})
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            manifest = output / "result-files-sha256.json"
            if manifest.exists():
                write_new(invocation / "previous-result-files-sha256.json", json.loads(manifest.read_text()))
            save_status(manifest, {str(p.relative_to(output)): file_hash(p)
                for p in sorted(output.rglob("*")) if p.is_file() and p != manifest})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=ALIASES)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--deployment", required=True, type=Path)
    p.add_argument("--deployment-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--phase", choices=("validation", "test", "both"), default="validation")
    p.add_argument("--thresholds", type=Path)
    p.add_argument("--thresholds-sha256")
    p.add_argument("--port", type=int, default=8093)
    p.add_argument("--authorization-note")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue the same identity/ledger within its cumulative three-attempt budget")
    args = p.parse_args(argv)
    if not args.execute:
        validate_inputs(args.bundle.resolve(), args.model, args.phase, args.deployment_sha256,
                        args.thresholds, args.thresholds_sha256)
        verify_deployment(args.deployment.resolve(), args.deployment_sha256, args.model)
        print(json.dumps({"offline_passed": True, "inference_calls": 0, "model": args.model}))
        return
    if not args.authorization_note:
        p.error("--execute requires the approved scope in --authorization-note")
    if not 1 <= args.port <= 65535:
        p.error("invalid port")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_XET="1")
    with (args.deployment / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args.model, args.bundle.resolve(), args.deployment.resolve(), args.output.resolve(),
                args.deployment_sha256, phase=args.phase, threshold_path=args.thresholds,
                threshold_hash=args.thresholds_sha256, port=args.port, authorization=args.authorization_note, resume=args.resume)




def validate_completion(output, split, cfg):
    if not (output / split / "completion.json").exists():
        return False
    _verify_completion(output, output / split,
                       calibration=split == "validation" and cfg["kind"] == "margin")
    return True


def selected_calibration(plan, cfg, ledger, bundle_digest, deployment_hash):
    selected = select_thresholds(plan, collect(plan, cfg, ledger), cfg["kind"], cfg["model"])
    selected.update(calibration="selected_on_complete_validation24", config_sha256=digest(cfg),
        rules_sha256=digest(RULES), bundle_sha256=bundle_digest, model=cfg["model"],
        native_deployment_sha256=deployment_hash)
    return selected


def validate_saved_path(path, output, *, directory=False):
    present = path.exists() or path.is_symlink()
    if (not path.resolve().is_relative_to(output.resolve())
            or present and not (path.is_dir() if directory else path.is_file())):
        raise ValueError("resume evidence path has wrong type or escapes output: " + str(path.relative_to(output)))


def preflight_resume(alias, bundle, output, cfg, bundle_digest, deployment_hash, phase, threshold_hash, sealed_receipts=None):
    """Inspect every saved phase before recovery, startup, or any output writes."""
    phases = ("validation", "test") if phase == "both" else (phase,)
    artifact = None
    if "validation" in phases and cfg["kind"] == "margin":
        validate_saved_path(output / "thresholds.json", output)
    if "validation" in phases and cfg["kind"] == "margin" and (output / "thresholds.json").exists():
        threshold_hash = file_hash(output / "thresholds.json")
        artifact = load_threshold_artifact(output / "thresholds.json", threshold_hash,
            bundle=bundle, alias=alias, config=cfg, bundle_digest=bundle_digest)
        if artifact.get("native_deployment_sha256") != deployment_hash:
            raise ValueError("threshold native deployment differs")
    for split in phases:
        plan = json.loads((bundle / "plans" / f"{alias}-{split}.json").read_text())
        context = context_for(plan, cfg, bundle_digest)
        phase_dir = output / split
        validate_saved_path(phase_dir, output, directory=True)
        validate_saved_path(phase_dir / "receipts", output, directory=True)
        for name in ("run.json", "attempts.jsonl", "completion.json", "report.json"):
            validate_saved_path(phase_dir / name, output)
        phase_record = phase_dir / "run.json"
        if phase_record.exists():
            existing = json.loads(phase_record.read_text())
            identity = {"context": context, "threshold_artifact_sha256": threshold_hash if split == "test" else None,
                        "native_deployment_sha256": deployment_hash}
            if not isinstance(existing, dict) or any(existing.get(k) != v for k, v in identity.items()):
                raise ValueError("resume phase identity differs")
        if (split == "test" and phase == "both" and cfg["kind"] == "margin"
                and phase_dir.exists() and artifact is None):
            raise ValueError("validation calibration missing; saved test phase blocked")
        completed = validate_completion(output, split, cfg)
        ledger = Ledger(phase_dir / "attempts.jsonl", context, read_only=True)
        try:
            pending = reconcile_receipts(phase_dir, ledger, plan, cfg, output, sealed_receipts, recover=False)
            if completed and pending:
                raise ValueError("completed phase has unrecorded receipts")
            # Project valid orphan evidence in memory only. Recovery, including
            # the first observation of a fatal receipt, still happens below.
            ledger.rows.extend(pending)
            if split == "validation" and artifact is not None:
                selected = selected_calibration(plan, cfg, ledger, bundle_digest, deployment_hash)
                if any(digest(artifact.get(k)) != digest(v) for k, v in selected.items()):
                    raise ValueError("saved calibration differs from validation ledger")
        finally:
            ledger.close()


if __name__ == "__main__":
    main()
