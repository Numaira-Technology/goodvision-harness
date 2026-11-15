"""Best-effort hardware and score log for the Qwen Vast launchers.

Writing a log must never change offer selection, the eval, or the process
exit code. Failures are printed and stored on the entry. Missing measurements
are the string ``unknown``, never a fabricated 0.

An idle GPU often reports PCIe gen 1, an SM clock around 210 MHz, and 0%
utilization. That is the normal power-saving state, not throttling. The max
link generation, the power limit, and the max SM clock are the capability
readings. ``host_*`` probe fields are the whole machine. ``container_*`` fields
are the cgroup limit of this rental, which is smaller than the host when the
offer's ``cpu_cores_effective`` and ``cpu_ram`` do not cover the machine.
"""
from __future__ import annotations

import atexit
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "goodvision.vast_run_log.v1"
OFFER_FIELDS = (
    "id", "machine_id", "host_id", "geolocation", "dph_total", "gpu_name",
    "num_gpus", "gpu_ram", "gpu_frac", "gpu_lanes", "pci_gen", "pcie_bw",
    "gpu_max_power", "cpu_name", "cpu_cores_effective", "cpu_ram", "inet_up",
    "inet_down", "disk_name", "disk_bw", "driver_version", "cuda_max_good",
    "reliability", "verification", "datacenter", "dlperf",
)
LAUNCHERS = {
    "start_qwen8b_r2.py": ("tools.start_qwen8b_r2", "qwen8b_r2_host", "transformers"),
    "start_qwen8b_vllm.py": ("tools.start_qwen8b_vllm", "qwen8b_r2_vllm", "vllm"),
}
_SECRET_KEY = re.compile(
    r"(?i)(secret|password|passwd|token|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|public[_-]?key|authorization|credential|ssh[_-]?key|"
    r"authorized_keys|id_rsa)$"
)
_ENV_KEY = re.compile(r"(?i)^(env|environment|environ|extra_env)$")
_SECRET_VALUE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"xox[baprs]-[A-Za-z0-9-]{8,}|ghp_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9\-]{20,}|"
    r"bearer\s+[A-Za-z0-9._\-]{8,})"
)
# One POSIX sh command. The controller may be Windows; the shell runs on the
# Linux instance over SSH. Host facts are a HOST-tab line so a CPU model that
# contains commas still parses. getconf and MemTotal are the whole machine.
# cpu.max / memory.max (cgroup v2, else v1) are this container's limit.
# Empty facts stay "unknown", never 0.
# Current PCIe gen and the SM clock fall while the GPU is idle. On the reviewed
# 4090 that was gen 1, 210 MHz, and 0% utilization: power-saving, not throttling.
_PROBE = (
    "nvidia-smi --query-gpu=name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,"
    "pcie.link.width.current,pcie.link.width.max,power.limit,clocks.sm,clocks.max.sm,"
    "clocks.mem,driver_version,memory.total,utilization.gpu --format=csv,noheader; "
    "smi_status=$?; "
    "cpu=$(awk -F: '/^model name/ {sub(/^[ \\t]+/, \"\", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null); "
    "cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true); "
    "ram=$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null); "
    "cuda=$(nvidia-smi -q 2>/dev/null | awk -F: '/^CUDA Version/ {sub(/^[ \\t]+/, \"\", $2); print $2; exit}'); "
    "if [ -r /sys/fs/cgroup/cpu.max ]; then "
    "container_cpu=\"v2 $(awk 'NR==1 { printf \"%s %s\", $1, $2; exit }' /sys/fs/cgroup/cpu.max)\"; "
    "elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then "
    "container_cpu=\"v1 $(awk 'NR==1 { printf \"%s\", $1; exit }' /sys/fs/cgroup/cpu/cpu.cfs_quota_us) "
    "$(awk 'NR==1 { printf \"%s\", $1; exit }' /sys/fs/cgroup/cpu/cpu.cfs_period_us)\"; "
    "elif [ -r /sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us ] && "
    "[ -r /sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us ]; then "
    "container_cpu=\"v1 $(awk 'NR==1 { printf \"%s\", $1; exit }' /sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us) "
    "$(awk 'NR==1 { printf \"%s\", $1; exit }' /sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us)\"; "
    "else container_cpu=unknown; fi; "
    "if [ -r /sys/fs/cgroup/memory.max ]; then "
    "container_mem=\"v2 $(awk 'NR==1 { printf \"%s\", $1; exit }' /sys/fs/cgroup/memory.max)\"; "
    "elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then "
    "container_mem=\"v1 $(awk 'NR==1 { printf \"%s\", $1; exit }' /sys/fs/cgroup/memory/memory.limit_in_bytes)\"; "
    "else container_mem=unknown; fi; "
    "[ -n \"$cpu\" ] || cpu=unknown; "
    "[ -n \"$cores\" ] || cores=unknown; "
    "if [ -n \"$ram\" ]; then ram=\"$ram kB\"; else ram=unknown; fi; "
    "[ -n \"$cuda\" ] || cuda=unknown; "
    "[ -n \"$container_cpu\" ] || container_cpu=unknown; "
    "[ -n \"$container_mem\" ] || container_mem=unknown; "
    "printf '\\nHOST\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "
    "\"$cpu\" \"$cores\" \"$ram\" \"$cuda\" \"$container_cpu\" \"$container_mem\"; "
    "exit \"$smi_status\""
)
_PROBE_COLUMNS = (
    "name", "pci_bus_id", "pcie_gen_current", "pcie_gen_max", "pcie_width_current",
    "pcie_width_max", "power_limit", "clock_sm", "clock_max_sm", "clock_mem",
    "driver_version", "memory_total", "utilization_gpu",
)
# First four match the HOST line from probes that predate container limits.
_HOST_LEGACY = ("cpu_model", "host_cpu_cores", "host_ram_total", "cuda_version")
_HOST_CONTAINER = ("container_cpu_limit", "container_memory_limit")
_HOST_FIELDS = _HOST_LEGACY + _HOST_CONTAINER
_IDLE_NOTE = (
    "Idle PCIe gen 1, a low SM clock such as 210 MHz, and 0% utilization are "
    "the GPU's normal power-saving state, not throttling. pcie_gen_max, "
    "power_limit, and clock_max_sm are the capability readings. host_cpu_cores "
    "and host_ram_total are the whole machine; container_cpu_limit and "
    "container_memory_limit are this rental's cgroup limit."
)
_SCORE_FIELDS = (
    "accuracy", "correct", "planned_questions", "complete_valid_questions",
    "incomplete_questions", "final_unknown_questions", "final_missing_unique_units",
)

_SESSION = None


def arm_if_launcher(filename: str) -> None:
    """Start a session when python -m is running a Qwen rental launcher."""
    base = Path(str(filename).replace("\\", "/")).name
    spec = LAUNCHERS.get(base)
    if spec is None:
        return
    global _SESSION
    if _SESSION is not None:
        return
    try:
        session = RunLog(*spec)
        session.install()
    except Exception as exc:
        _stderr("arm", exc)
        return
    _SESSION = session
    atexit.register(session.finish_once)


def redact(value):
    """Return a JSON-safe copy with secret-looking keys and values removed."""
    return _redact(_jsonable(value))


def redact_text(text: str) -> str:
    return _SECRET_VALUE.sub("redacted", str(text))[:300]


def offer_fields(offer) -> dict:
    if not isinstance(offer, dict):
        return {name: "unknown" for name in OFFER_FIELDS}
    out = {}
    for name in OFFER_FIELDS:
        out[name] = _measurement(offer.get(name)) if name in offer else "unknown"
    return out


def parse_probe(text: str) -> dict:
    """Parse nvidia-smi CSV rows plus optional HOST lines.

    Lines that are not a GPU row or a HOST record are ignored. One banner
    line must not drop a good reading. A probe with no GPU row is unknown,
    not 0. A usable GPU table with no host line keeps the GPUs and marks
    each host fact unknown.
    """
    rows = []
    host = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("HOST\t") or stripped == "HOST":
            parsed = _parse_host_line(stripped)
            if parsed is not None:
                host = parsed
            continue
        cells = next(csv.reader([line]))
        cells = [cell.strip().replace("\r", "") for cell in cells]
        if len(cells) != len(_PROBE_COLUMNS):
            continue
        rows.append(dict(zip(_PROBE_COLUMNS, cells)))
    if not rows:
        return _unknown_probe(text)
    return {
        "status": "ok",
        "gpus": rows,
        "host": host if host is not None else _unknown_host(),
        "note": _IDLE_NOTE,
    }


def _unknown_host() -> dict:
    return {name: "unknown" for name in _HOST_FIELDS}


def _unknown_probe(text: str) -> dict:
    return {"status": "unknown", "gpus": "unknown", "host": "unknown", "raw": redact_text(text)}


def _parse_host_line(line: str):
    parts = line.split("\t")
    if not parts or parts[0].strip() != "HOST":
        return None
    values = parts[1:]
    if len(values) == len(_HOST_LEGACY):
        names = _HOST_LEGACY
    elif len(values) == len(_HOST_FIELDS):
        names = _HOST_FIELDS
    else:
        return None
    parsed = _unknown_host()
    for name, value in zip(names, values):
        text = value.replace("\r", "").strip()
        if not text:
            continue
        if name == "container_cpu_limit":
            text = _format_container_cpu(text)
        elif name == "container_memory_limit":
            text = _format_container_memory(text)
        parsed[name] = text
    return parsed


def _cores_from_quota(quota: str, period: str) -> str:
    if not quota.isdigit() or not period.isdigit():
        return "unknown"
    q, p = int(quota), int(period)
    if p == 0:
        return "unknown"
    whole, rem = divmod(q, p)
    if rem == 0:
        return str(whole)
    scaled = (q * 1000) // p
    text = "%d.%03d" % divmod(scaled, 1000)
    return text.rstrip("0")


def _format_container_cpu(raw: str) -> str:
    parts = raw.split()
    if len(parts) != 3 or parts[0] not in {"v1", "v2"}:
        return "unknown"
    kind, quota, period = parts
    source = "cgroup v2 cpu.max" if kind == "v2" else "cgroup v1 cpu.cfs_quota_us"
    if quota in {"max", "-1"}:
        return "max (%s)" % source
    cores = _cores_from_quota(quota, period)
    if cores == "unknown":
        return "unknown"
    return "%s cores (%s %s %s)" % (cores, source, quota, period)


def _format_container_memory(raw: str) -> str:
    parts = raw.split()
    if len(parts) != 2 or parts[0] not in {"v1", "v2"}:
        return "unknown"
    kind, value = parts
    source = "cgroup v2 memory.max" if kind == "v2" else "cgroup v1 memory.limit_in_bytes"
    if value == "max":
        return "max (%s)" % source
    if not value.isdigit():
        return "unknown"
    number = int(value)
    if kind == "v1" and number >= 1 << 62:
        return "max (%s)" % source
    return "%d B (%s)" % (number, source)


def summarize_score(eval_output, alias) -> dict | str:
    if not eval_output:
        return "unknown"
    base = Path(eval_output) / alias
    if not base.is_dir():
        return "unknown"
    found = False
    summary = {}
    for split in ("validation", "test"):
        path = base / split / "summary.json"
        if not path.is_file():
            summary[split] = "unknown"
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary[split] = "unknown"
            continue
        found = True
        summary[split] = _score_split(payload if isinstance(payload, dict) else {})
    manifest = Path(eval_output) / "run-manifest.json"
    slots = "unknown"
    if manifest.is_file():
        try:
            body = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(body, dict) and "slots_per_model" in body:
                slots = _measurement(body.get("slots_per_model"))
        except (OSError, ValueError):
            slots = "unknown"
    if not found and slots == "unknown":
        return "unknown"
    summary["concurrency_slots"] = slots
    return summary


def short_score(score) -> dict | str:
    if not isinstance(score, dict):
        return "unknown"
    out = {}
    for split in ("validation", "test"):
        part = score.get(split)
        if not isinstance(part, dict):
            out[split] = "unknown"
            continue
        out[split] = {
            name: part.get(name, "unknown")
            for name in ("accuracy", "final_unknown_questions", "final_missing_unique_units")
        }
    return out


def logs_directory(alias: str, eval_output=None) -> Path:
    """Stable log root. Never a directory the eval output mkdir must create."""
    override = os.environ.get("GOODVISION_RUN_LOG_ROOT", "").strip()
    if override:
        root = Path(override) / alias / "logs"
    else:
        root = REPO / "outputs" / "run-logs" / alias / "logs"
    if not eval_output:
        return root
    return _avoid_eval_output(root, Path(eval_output))


class RunLog:
    def __init__(self, launcher: str, model: str, engine: str):
        argv_model = _flag("--model")
        self.launcher = launcher
        self.model = argv_model or model
        self.engine = engine
        self.eval_output = _flag("--eval-output")
        phase = _flag("--eval-phase")
        self.eval_phase = phase or "both"
        self.eval_phase_source = "argv" if phase else "launcher-default"
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + os.urandom(4).hex()
        self.root = logs_directory(self.model, self.eval_output)
        self.run_dir = self.root / self.run_id
        self.who = _who()
        self.git_commit = _git_commit()
        self.logging_errors = []
        self.offer = None
        self.instance_id = None
        self.probe = "unknown"
        self.score = "unknown"
        self.eval_returncode = "unknown"
        self._mono = {}
        self._timings = {name: "unknown" for name in (
            "rent_start_utc", "ready_utc", "eval_start_utc", "eval_end_utc", "release_utc",
        )}
        self._probed = False
        self._hooks_ready = False
        self._finished = False
        self._originals = {}

    def install(self) -> None:
        import tools.start_sam as sam
        self._originals = {
            "create_from_offers": sam.create_from_offers,
            "run": sam.run,
            "release_instance": sam.release_instance,
        }
        sam.create_from_offers = _wrap_create(sam.create_from_offers, self)
        sam.run = _wrap_run(sam.run, self)
        sam.release_instance = _wrap_release(sam.release_instance, self)

    def note_create(self, offers, args, kwargs, created) -> None:
        contract = _contract(created)
        if contract is None:
            return
        self.instance_id = contract
        offer_id = kwargs.get("id", args[0] if args else None)
        selected = _find_offer(offers, offer_id)
        self.offer = selected
        self._write_offer()
        self._write_entry()
        self._update_manifest()
        _stderr_line("run log: saved offer %s" % _display(self.run_dir / "vast-offer.json"))

    def mark_rent_start(self) -> None:
        if self._timings["rent_start_utc"] == "unknown":
            self._timings["rent_start_utc"] = _utc()
            self._mono["rent_start"] = time.monotonic()

    def note_ready(self) -> None:
        self._timings["ready_utc"] = _utc()
        self._mono["ready"] = time.monotonic()

    def note_eval_start(self) -> None:
        self._timings["eval_start_utc"] = _utc()
        self._mono["eval_start"] = time.monotonic()

    def note_eval_end(self, code) -> None:
        self._timings["eval_end_utc"] = _utc()
        self._mono["eval_end"] = time.monotonic()
        self.eval_returncode = code if isinstance(code, int) else "unknown"

    def note_release(self) -> None:
        self._timings["release_utc"] = _utc()
        self._mono["release"] = time.monotonic()

    def arm_stage_hooks(self, caller_globals: dict) -> None:
        if self._hooks_ready:
            return
        self._hooks_ready = True
        fn = caller_globals.get("_wait_ready")
        if callable(fn) and not getattr(fn, "_gv_wrapped", False):
            caller_globals["_wait_ready"] = _wrap_ready(fn, self)
        try:
            import tools.qwen8b.eval_local as eval_local
        except Exception as exc:
            sys.modules.pop("tools.qwen8b.eval_local", None)
            self._error("eval hook", exc)
            return
        current = eval_local.main
        if not getattr(current, "_gv_wrapped", False):
            eval_local.main = _wrap_eval(current, self)

    def maybe_probe(self, command) -> None:
        if self._probed or not isinstance(command, (list, tuple)) or not command:
            return
        program = str(command[0]).replace("\\", "/").rsplit("/", 1)[-1]
        if program != "ssh" or len(command) < 2:
            return
        self._probed = True
        prefix = list(command[:-1])
        try:
            result = subprocess.run(
                [*prefix, _PROBE], capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.probe = "unknown"
            self._error("probe", exc)
            return
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if result.returncode != 0:
            self.probe = {
                "status": "unknown", "gpus": "unknown", "host": "unknown",
                "raw": redact_text(stdout + ("\n" + stderr if stderr else "")),
            }
            self._error("probe", "nvidia-smi exited %s" % result.returncode)
            return
        self.probe = parse_probe(stdout)
        if stderr:
            self.probe["raw"] = redact_text(stderr)
        self._write_entry()

    def capture_score(self) -> None:
        try:
            self.score = summarize_score(self.eval_output, self.model)
        except Exception as exc:
            self.score = "unknown"
            self._error("score", exc)

    def finish_once(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._mono["finish"] = time.monotonic()
        try:
            self.capture_score()
            self._write_entry()
            self._update_manifest()
            self._mirror_report()
            _stderr_line("run log: %s" % _display(self.run_dir / "entry.json"))
            if self.logging_errors:
                _stderr_line("run log: %s warning(s); eval was not changed" % len(self.logging_errors))
        except Exception as exc:
            _stderr("finish", exc)

    def _write_offer(self) -> None:
        if self.offer is None:
            return
        path = self.run_dir / "vast-offer.json"
        if path.exists():
            return
        _write_new(path, redact(self.offer))

    def _entry(self) -> dict:
        return {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "launcher": self.launcher,
            "model": self.model,
            "engine": self.engine,
            "eval_output": self.eval_output or "unknown",
            "eval_phase": self.eval_phase,
            "eval_phase_source": self.eval_phase_source,
            "instance_id": _measurement(self.instance_id) if self.instance_id is not None else "unknown",
            "who": self.who,
            "git_commit": self.git_commit,
            "timings": self._timing_block(),
            "hardware": {"offer": offer_fields(self.offer), "probe": self.probe},
            "score": self.score,
            "eval_returncode": self.eval_returncode,
            "logging_errors": list(self.logging_errors),
        }

    def _timing_block(self) -> dict:
        block = dict(self._timings)
        start = self._mono.get("rent_start")
        end = self._mono.get("release", self._mono.get("eval_end", self._mono.get("finish")))
        if start is None or end is None:
            block["wall_seconds"] = "unknown"
        else:
            block["wall_seconds"] = round(end - start, 3)
        return block

    def _row(self) -> dict:
        fields = offer_fields(self.offer)
        return {
            "run_id": self.run_id,
            "time": self._timings["rent_start_utc"] if self._timings["rent_start_utc"] != "unknown" else _utc(),
            "who": self.who,
            "gpu": {
                "machine_id": fields["machine_id"],
                "gpu_name": fields["gpu_name"],
                "dph_total": fields["dph_total"],
                "geolocation": fields["geolocation"],
            },
            "score": short_score(self.score),
            "entry": self.run_id + "/entry.json",
        }

    def _write_entry(self) -> None:
        _atomic_write(self.run_dir / "entry.json", self._entry())

    def _update_manifest(self) -> None:
        _upsert_manifest(self.root / "manifest.json", self._row(), self)

    def _mirror_report(self) -> None:
        if not self.eval_output:
            return
        report = Path(self.eval_output) / self.model
        if not report.is_dir():
            return
        dest = report / "logs" / self.run_id
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("vast-offer.json", "entry.json"):
            src = self.run_dir / name
            if src.is_file():
                _atomic_write(dest / name, json.loads(src.read_text(encoding="utf-8")))
        _upsert_manifest(report / "logs" / "manifest.json", self._row(), self)

    def _error(self, label, exc) -> None:
        message = "%s failed (%s)" % (label, type(exc).__name__ if isinstance(exc, BaseException) else "error")
        detail = redact_text(exc if isinstance(exc, str) else exc)
        if detail:
            message = "%s: %s" % (message, detail)
        self.logging_errors.append(message)
        _stderr_line("run log: " + message)


def _wrap_create(original, session: RunLog):
    def wrapper(vast, offers, *args, **kwargs):
        try:
            session.mark_rent_start()
        except Exception as exc:
            session._error("timer", exc)
        real = getattr(vast, "create_instance", None)
        if not callable(real):
            return original(vast, offers, *args, **kwargs)

        def spy(*spy_args, **spy_kwargs):
            try:
                session.mark_rent_start()
            except Exception as exc:
                session._error("timer", exc)
            result = real(*spy_args, **spy_kwargs)
            try:
                session.note_create(offers, spy_args, spy_kwargs, result)
            except Exception as exc:
                session._error("offer", exc)
            return result

        try:
            vast.create_instance = spy
        except Exception as exc:
            session._error("offer", exc)
            return original(vast, offers, *args, **kwargs)
        try:
            return original(vast, offers, *args, **kwargs)
        finally:
            try:
                vast.create_instance = real
            except Exception as exc:
                session._error("offer", exc)

    wrapper._gv_wrapped = True
    return wrapper


def _wrap_run(original, session: RunLog):
    def wrapper(command, *, input=None, timeout=120):
        try:
            session.arm_stage_hooks(sys._getframe(1).f_globals)
        except Exception as exc:
            session._error("hook", exc)
        result = original(command, input=input, timeout=timeout)
        try:
            session.maybe_probe(command)
        except Exception as exc:
            session._error("probe", exc)
        return result

    wrapper._gv_wrapped = True
    return wrapper


def _wrap_release(original, session: RunLog):
    def wrapper(vast, instance_id):
        try:
            return original(vast, instance_id)
        finally:
            try:
                session.note_release()
            except Exception as exc:
                session._error("release", exc)

    wrapper._gv_wrapped = True
    return wrapper


def _wrap_ready(fn, session: RunLog):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        try:
            session.note_ready()
        except Exception as exc:
            session._error("ready", exc)
        return result

    wrapper._gv_wrapped = True
    return wrapper


def _wrap_eval(fn, session: RunLog):
    def wrapper(*args, **kwargs):
        try:
            session.note_eval_start()
        except Exception as exc:
            session._error("timer", exc)
        try:
            code = fn(*args, **kwargs)
        except Exception:
            try:
                session.note_eval_end("unknown")
            except Exception as exc:
                session._error("timer", exc)
            raise
        try:
            session.note_eval_end(code)
            session.capture_score()
            session._write_entry()
        except Exception as exc:
            session._error("score", exc)
        return code

    wrapper._gv_wrapped = True
    return wrapper


def _measurement(value):
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return "unknown"
    if isinstance(value, (int, float, str)):
        return value
    return "unknown"


def _score_split(payload: dict) -> dict:
    out = {}
    for name in _SCORE_FIELDS:
        out[name] = _measurement(payload.get(name)) if name in payload else "unknown"
    latency = payload.get("test_attempt_latency")
    if isinstance(latency, dict):
        out["p50_seconds"] = _measurement(latency.get("p50_seconds")) if "p50_seconds" in latency else "unknown"
        out["p95_seconds"] = _measurement(latency.get("p95_seconds")) if "p95_seconds" in latency else "unknown"
        out["latency_n"] = _measurement(latency.get("n")) if "n" in latency else "unknown"
        out["latency_source"] = latency.get("source") if isinstance(latency.get("source"), str) else "unknown"
    else:
        out["p50_seconds"] = "unknown"
        out["p95_seconds"] = "unknown"
        out["latency_n"] = "unknown"
        out["latency_source"] = "unknown"
    return out


def _find_offer(offers, offer_id):
    if not isinstance(offers, list):
        return None
    for offer in offers:
        if isinstance(offer, dict) and _same_id(offer.get("id"), offer_id):
            return offer
    return None


def _same_id(left, right) -> bool:
    try:
        return int(left) == int(right) and not isinstance(left, bool) and not isinstance(right, bool)
    except (TypeError, ValueError):
        return False


def _contract(created):
    if not isinstance(created, dict):
        return None
    contract = created.get("new_contract")
    if (
        created.get("success") is True
        and isinstance(contract, int)
        and not isinstance(contract, bool)
        and contract > 0
    ):
        return contract
    return None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "unknown"
    if isinstance(value, str) or value is None:
        return value
    return str(value)


def _redact(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if _ENV_KEY.match(str(key)) or _SECRET_KEY.search(str(key)):
                cleaned[key] = "redacted"
            else:
                cleaned[key] = _redact(item)
        return cleaned
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return "redacted" if _SECRET_VALUE.search(value) else value
    return value


def _who() -> dict:
    operator = os.environ.get("GOODVISION_OPERATOR", "").strip() or "unknown"
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    git_user = _git_user()
    if operator != "unknown":
        display = operator
    elif git_user != "unknown":
        display = git_user
    else:
        display = user
    return {"display": display, "operator_env": operator, "user": user, "git_user": git_user}


def _git_user() -> str:
    return _git_line(["config", "--get", "user.name"])


def _git_commit() -> str:
    return _git_line(["rev-parse", "HEAD"])


def _git_line(args) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    text = (result.stdout or "").strip()
    return text or "unknown"


def _flag(name: str):
    argv = sys.argv
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        prefix = name + "="
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_dumps(value))
    os.replace(tmp, path)


def _write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(_dumps(value))


def _upsert_manifest(path: Path, row: dict, session: RunLog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(".manifest.lock")
    acquired = _acquire_lock(lock)
    try:
        rows = []
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                session._error("manifest", exc)
                return
            if not isinstance(loaded, list):
                session._error("manifest", "manifest is not a list")
                return
            rows = [item for item in loaded if not (isinstance(item, dict) and item.get("run_id") == row["run_id"])]
        rows.append(row)
        _atomic_write(path, rows)
    finally:
        if acquired:
            try:
                lock.unlink()
            except OSError:
                pass


def _acquire_lock(lock: Path) -> bool:
    for _ in range(40):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 30:
                    lock.unlink()
                    continue
            except OSError:
                pass
            time.sleep(0.05)
    return False


def _avoid_eval_output(root: Path, eval_output: Path) -> Path:
    try:
        root_abs = root.resolve()
        ev_abs = eval_output.resolve()
    except OSError:
        return root
    if root_abs == ev_abs or root_abs.is_relative_to(ev_abs):
        return Path(str(eval_output) + ".run-logs") / "logs"
    return root


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _stderr(label: str, exc: BaseException) -> None:
    _stderr_line("run log: %s failed (%s); eval continues" % (label, type(exc).__name__))


def _stderr_line(text: str) -> None:
    sys.stderr.write(redact_text(text) + "\n")
    sys.stderr.flush()
