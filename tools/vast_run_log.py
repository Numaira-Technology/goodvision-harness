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
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "goodvision.vast_run_log.v1"
OFFER_FIELDS = (
    "id",
    "machine_id",
    "host_id",
    "geolocation",
    "dph_total",
    "gpu_name",
    "num_gpus",
    "gpu_ram",
    "gpu_frac",
    "gpu_lanes",
    "pci_gen",
    "pcie_bw",
    "gpu_max_power",
    "cpu_name",
    "cpu_cores_effective",
    "cpu_ram",
    "inet_up",
    "inet_down",
    "disk_name",
    "disk_bw",
    "driver_version",
    "cuda_max_good",
    "reliability",
    "verification",
    "datacenter",
    "dlperf",
)
# module, model alias, engine, sample-during-run.
# The HF launcher keeps the offer log and does not sample.
LAUNCHERS = {
    "start_qwen8b_r2.py": (
        "tools.start_qwen8b_r2",
        "qwen8b_r2_host",
        "transformers",
        False,
    ),
    "start_qwen8b_vllm.py": ("tools.start_qwen8b_vllm", "qwen8b_r2_vllm", "vllm", True),
    "start_sam.py": ("tools.start_sam", "sam", "sam", True),
    "start_detector.py": ("tools.start_detector", "detector", "detector", True),
    "diag_qwen8b_vllm_import.py": (
        "tools.diag_qwen8b_vllm_import",
        "qwen8b_vllm_diag",
        "vllm-diag",
        True,
    ),
}
# Armed by tools/__init__.py because that launcher file must not be edited.
_AUTO_ARM = {"start_qwen8b_r2.py"}
SAMPLE_INTERVAL = 2
_SAMPLE_COLUMNS = (
    "pcie_gen_current",
    "pcie_width_current",
    "pcie_gen_max",
    "pcie_width_max",
    "power_draw",
    "power_limit",
    "clock_sm",
    "utilization_gpu",
    "memory_used",
    "temperature_gpu",
)
_STATIC_FIELDS = (
    "driver_version",
    "cuda_version",
    "cpu_model",
    "host_cpu_cores",
    "host_ram_total",
    "container_cpu_limit",
    "container_memory_limit",
)
# POSIX sh on the instance. No pip. One nvidia-smi line every two seconds.
_SAMPLER_SH = (
    "out=$1\n"
    "interval=$2\n"
    'exec >> "$out" || exit 1\n'
    'trap "exit 0" TERM INT\n'
    "while true; do\n"
    "  ts=$(date -u +%s 2>/dev/null || printf unknown)\n"
    "  nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current,"
    "pcie.link.gen.max,pcie.link.width.max,power.draw,power.limit,clocks.sm,"
    "utilization.gpu,memory.used,temperature.gpu --format=csv,noheader "
    '2>/dev/null | awk -v ts="$ts" \'NF { print ts "\\t" $0 }\'\n'
    '  sleep "$interval"\n'
    "done\n"
)
_SECRET_KEY = re.compile(
    r"(?i)(secret|password|passwd|token|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|public[_-]?key|authorization|credential|ssh[_-]?key|"
    r"authorized_keys|id_rsa)(?:s|_id)?$"
)
_ENV_KEY = re.compile(r"(?i)^(env|environment|environ|extra_env)$")
_SECRET_VALUE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"xox[baprs]-[A-Za-z0-9-]{8,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9\-]{20,}|"
    r"(?:bearer|basic)\s+[A-Za-z0-9+/=._\-]{8,})"
)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
    r"(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:[\w-]*(?:secret|password|passwd|token|api[_-]?key|access[_-]?key|"
    r"credential|authorization)[\w-]*[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s&,;]+)"
)
_URL_CREDENTIAL = re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@")
# One POSIX sh command. The controller may be Windows; the shell runs on the
# Linux instance over SSH. Host facts are a HOST-tab line so a CPU model that
# contains commas still parses. getconf and MemTotal are the whole machine.
# cpu.max / memory.max (cgroup v2, else v1) are this container's limit.
# Empty facts stay "unknown", never 0.
# Current PCIe gen and the SM clock fall while the GPU is idle. On the reviewed
# 4090 that was gen 1, 210 MHz, and 0% utilization: power-saving, not throttling.
_PROBE = (
    "nvidia-smi "
    "--query-gpu=name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,"
    "pcie.link.width.current,pcie.link.width.max,power.limit,cloc"
    "ks.sm,clocks.max.sm,"
    "clocks.mem,driver_version,memory.total,utilization.gpu --format=csv,noheader; "
    "smi_status=$?; "
    'cpu=$(awk -F: \'/^model name/ {sub(/^[ \\t]+/, "", $2); print '
    "$2; exit}' /proc/cpuinfo 2>/dev/null); "
    "cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true); "
    "ram=$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null); "
    "cuda=$(nvidia-smi -q 2>/dev/null | awk -F: '/^CUDA Version/ "
    '{sub(/^[ \\t]+/, "", $2); print $2; exit}\'); '
    "if [ -r /sys/fs/cgroup/cpu.max ]; then "
    'container_cpu="v2 $(awk \'NR==1 { printf "%s %s", $1, $2; '
    "exit }' /sys/fs/cgroup/cpu.max)\"; "
    "elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r "
    "/sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then "
    'container_cpu="v1 $(awk \'NR==1 { printf "%s", $1; exit }\' '
    "/sys/fs/cgroup/cpu/cpu.cfs_quota_us) "
    "$(awk 'NR==1 { printf \"%s\", $1; exit }' "
    '/sys/fs/cgroup/cpu/cpu.cfs_period_us)"; '
    "elif [ -r /sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us ] && "
    "[ -r /sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us ]; then "
    'container_cpu="v1 $(awk \'NR==1 { printf "%s", $1; exit }\' '
    "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us) "
    "$(awk 'NR==1 { printf \"%s\", $1; exit }' "
    '/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us)"; '
    "else container_cpu=unknown; fi; "
    "if [ -r /sys/fs/cgroup/memory.max ]; then "
    'container_mem="v2 $(awk \'NR==1 { printf "%s", $1; exit }\' '
    '/sys/fs/cgroup/memory.max)"; '
    "elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then "
    'container_mem="v1 $(awk \'NR==1 { printf "%s", $1; exit }\' '
    '/sys/fs/cgroup/memory/memory.limit_in_bytes)"; '
    "else container_mem=unknown; fi; "
    '[ -n "$cpu" ] || cpu=unknown; '
    '[ -n "$cores" ] || cores=unknown; '
    'if [ -n "$ram" ]; then ram="$ram kB"; else ram=unknown; fi; '
    '[ -n "$cuda" ] || cuda=unknown; '
    '[ -n "$container_cpu" ] || container_cpu=unknown; '
    '[ -n "$container_mem" ] || container_mem=unknown; '
    "printf '\\nHOST\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "
    '"$cpu" "$cores" "$ram" "$cuda" "$container_cpu" "$container_mem"; '
    "gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor "
    "2>/dev/null || true); "
    '[ -n "$gov" ] || gov=unknown; '
    "turbo=unknown; "
    "if [ -r /sys/devices/system/cpu/intel_pstate/no_turbo ]; then "
    "nt=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true); "
    'if [ "$nt" = 0 ]; then turbo=enabled; '
    'elif [ "$nt" = 1 ]; then turbo=disabled; fi; '
    "elif [ -r /sys/devices/system/cpu/cpufreq/boost ]; then "
    "b=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true); "
    'if [ "$b" = 1 ]; then turbo=enabled; '
    'elif [ "$b" = 0 ]; then turbo=disabled; fi; '
    "fi; "
    "numa_nodes=$(find /sys/devices/system/node -maxdepth 1 -type d "
    "-name 'node[0-9]*' 2>/dev/null | wc -l | awk '{print $1}'); "
    '[ -n "$numa_nodes" ] || numa_nodes=unknown; '
    "gpu_numa=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader "
    "2>/dev/null | awk -F, '{"
    'gsub(/^[ \\t]+|[ \\t]+$/, "", $1); '
    'gsub(/^[ \\t]+|[ \\t]+$/, "", $2); '
    "bus=tolower($2); sub(/^00000000:/, \"0000:\", bus); "
    'node="unknown"; path="/sys/bus/pci/devices/" bus "/numa_node"; '
    "if ((getline n < path) > 0) { node=n; close(path) } "
    'if (out != "") out=out ";"; '
    'out=out $1 "@" bus "=" node '
    '} END { if (out == "") print "unknown"; else print out }\'); '
    '[ -n "$gpu_numa" ] || gpu_numa=unknown; '
    "apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,"
    "used_gpu_memory --format=csv,noheader 2>/dev/null | awk -F, '{"
    'gsub(/^[ \\t]+|[ \\t]+$/, "", $1); '
    'gsub(/^[ \\t]+|[ \\t]+$/, "", $2); '
    'gsub(/^[ \\t]+|[ \\t]+$/, "", $3); '
    'gsub(/^[ \\t]+|[ \\t]+$/, "", $4); '
    'name=$3; gsub(/[^A-Za-z0-9._+-]/, "", name); '
    'if (name == "") name="unknown"; '
    'if (out != "") out=out ";"; '
    'out=out "pid=" $2 ",name=" name ",mem=" $4 '
    '} END { if (out == "") print "none"; else print out }\'); '
    '[ -n "$apps" ] || apps=none; '
    "printf 'EXTRA\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "
    '"$gov" "$turbo" "$numa_nodes" "$gpu_numa" "$apps"; '
    'exit "$smi_status"'
)

_PROBE_COLUMNS = (
    "name",
    "pci_bus_id",
    "pcie_gen_current",
    "pcie_gen_max",
    "pcie_width_current",
    "pcie_width_max",
    "power_limit",
    "clock_sm",
    "clock_max_sm",
    "clock_mem",
    "driver_version",
    "memory_total",
    "utilization_gpu",
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
    "accuracy",
    "correct",
    "planned_questions",
    "complete_valid_questions",
    "incomplete_questions",
    "final_unknown_questions",
    "final_missing_unique_units",
)

_SESSION = None


def arm_if_launcher(filename: str, *, late: bool = False) -> None:
    """Start a session for the HF launcher. Other renters call ``attach``."""
    base = Path(str(filename).replace("\\", "/")).name
    spec = LAUNCHERS.get(base)
    if spec is None:
        return
    if not late and base not in _AUTO_ARM:
        return
    global _SESSION
    if _SESSION is not None:
        return
    try:
        session = RunLog(*spec)
        if base in _AUTO_ARM:
            session.install()
    except Exception as exc:
        _stderr("arm", exc)
        return
    _SESSION = session
    atexit.register(session.finish_once)


def attach(globals_dict=None) -> None:
    """Record this Vast rental. Call once at the start of ``main``.

    The call is best-effort: a failure is printed and does not change the
    rental, the eval, offer selection, or the process exit code. The running
    file must be a key of ``LAUNCHERS``. ``start_qwen8b_r2.py`` is the
    exception armed by ``tools/__init__.py`` and must not grow this call.
    """
    if globals_dict is None:
        globals_dict = sys._getframe(1).f_globals
    filename = str(globals_dict.get("__file__", ""))
    try:
        arm_if_launcher(filename, late=True)
    except Exception as exc:
        _stderr("arm", exc)
        return
    session = _SESSION
    base = Path(filename.replace("\\", "/")).name
    if session is None or base not in LAUNCHERS or base in _AUTO_ARM:
        return
    try:
        session.install()
    except Exception as exc:
        session._error("arm", exc)
    try:
        _patch_namespace(globals_dict, session)
    except Exception as exc:
        session._error("arm", exc)


def record_setup(name: str, seconds) -> None:
    """Store a setup duration on the open session. Never raises."""
    if _SESSION is None:
        return
    try:
        _SESSION.note_setup(name, seconds)
    except Exception as exc:
        _SESSION._error("setup", exc)


def record_client_access(facts) -> None:
    """Store client RTT and SSH-tunnel versus direct access. Never raises."""
    if _SESSION is None:
        return
    try:
        _SESSION.client_access = facts if isinstance(facts, dict) else "unknown"
        _SESSION._write_entry()
    except Exception as exc:
        _SESSION._error("client-access", exc)


def record_metrics_diff(diff) -> None:
    """Store a vLLM /metrics before/after diff. Never raises."""
    if _SESSION is None:
        return
    try:
        _SESSION.metrics_diff = diff if isinstance(diff, dict) else "unknown"
        if isinstance(diff, dict):
            _atomic_write(_SESSION.run_dir / "metrics-diff.json", diff)
        _SESSION._write_entry()
    except Exception as exc:
        _SESSION._error("metrics", exc)


def finish() -> None:
    """Flush while SSH credentials still exist, including a retained rental."""
    if _SESSION is not None:
        try:
            _SESSION.finish_once()
        except Exception as exc:
            _stderr("finish", exc)


def _mocked(fn) -> bool:
    return getattr(type(fn), "__module__", "") == "unittest.mock"


def _patch_namespace(mapping, session: RunLog) -> None:
    wrappers = (
        ("create_from_offers", lambda fn: _wrap_create(fn, session)),
        ("run", lambda fn: _wrap_run(fn, session)),
        ("release_instance", lambda fn: _wrap_release(fn, session)),
        ("_wait_ready", lambda fn: _wrap_ready(fn, session)),
        ("wait_for_worker", lambda fn: _wrap_ready(fn, session)),
    )
    for name, wrap in wrappers:
        fn = mapping.get(name)
        if not callable(fn) or _mocked(fn) or getattr(fn, "_gv_wrapped", False):
            continue
        mapping[name] = wrap(fn)
        if name in {"_wait_ready", "wait_for_worker"}:
            session._expects_ready = True


def redact(value):
    """Return a JSON-safe copy with secret-looking keys and values removed."""
    return _redact(_jsonable(value))


def redact_text(text: str) -> str:
    return _sanitize_text(str(text))[:300]


def _sanitize_text(text: str) -> str:
    text = _PRIVATE_KEY.sub("redacted", text)
    text = _SECRET_VALUE.sub("redacted", text)
    text = _SECRET_ASSIGNMENT.sub("redacted", text)
    return _URL_CREDENTIAL.sub(r"\1redacted@", text)


def offer_fields(offer) -> dict:
    if not isinstance(offer, dict):
        return {name: "unknown" for name in OFFER_FIELDS}
    out = {}
    for name in OFFER_FIELDS:
        out[name] = _measurement(offer.get(name)) if name in offer else "unknown"
    return out


def parse_probe(text: str) -> dict:
    """Parse nvidia-smi CSV rows plus optional HOST lines.

    Lines that are not a GPU row, a HOST record, or an EXTRA record are
    ignored. One banner line must not drop a good reading. A probe with no
    GPU row is unknown, not 0. A usable GPU table with no host line keeps
    the GPUs and marks each host fact unknown. EXTRA is the investigation
    facts (governor, turbo, NUMA, GPU NUMA node, compute apps) and is
    omitted when the probe predates that line.
    """
    rows = []
    host = None
    investigation = None
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("EXTRA\t") or stripped == "EXTRA":
            parsed = _parse_extra_line(line.lstrip(" ").rstrip("\r"))
            if parsed is not None:
                investigation = parsed
            continue
        if stripped.startswith("HOST\t") or stripped == "HOST":
            parsed = _parse_host_line(line.lstrip(" ").rstrip("\r"))
            if parsed is not None:
                host = parsed
            continue
        cells = next(csv.reader([line]))
        cells = [cell.strip().replace("\r", "") for cell in cells]
        if len(cells) != len(_PROBE_COLUMNS) or not re.fullmatch(
            r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", cells[1]
        ):
            continue
        cells = [
            "unknown"
            if cell.lower() in {"", "n/a", "[n/a]", "not supported", "[not supported]"}
            else cell
            for cell in cells
        ]
        rows.append(dict(zip(_PROBE_COLUMNS, cells, strict=True)))
    if not rows:
        return _unknown_probe(text)
    parsed = {
        "status": "ok",
        "gpus": rows,
        "host": host if host is not None else _unknown_host(),
        "note": _IDLE_NOTE,
    }
    if investigation is not None:
        parsed["investigation"] = investigation
    return parsed


_EXTRA_FIELDS = (
    "cpu_governor",
    "cpu_turbo",
    "numa_node_count",
    "gpu_numa_nodes",
    "gpu_compute_apps",
)


def _parse_extra_line(line: str):
    parts = line.split("\t")
    if not parts or parts[0].strip() != "EXTRA" or len(parts) != 1 + len(_EXTRA_FIELDS):
        return None
    parsed = {}
    for name, value in zip(_EXTRA_FIELDS, parts[1:], strict=True):
        text = value.replace("\r", "").strip()
        parsed[name] = text or "unknown"
    return parsed


def _unknown_host() -> dict:
    return {name: "unknown" for name in _HOST_FIELDS}


def _unknown_probe(text: str) -> dict:
    return {
        "status": "unknown",
        "gpus": "unknown",
        "host": "unknown",
        "raw": redact_text(text),
    }


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
    for name, value in zip(names, values, strict=True):
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
    text = "{}.{:03d}".format(*divmod(scaled, 1000))
    return text.rstrip("0")


def _format_container_cpu(raw: str) -> str:
    parts = raw.split()
    if len(parts) != 3 or parts[0] not in {"v1", "v2"}:
        return "unknown"
    kind, quota, period = parts
    source = "cgroup v2 cpu.max" if kind == "v2" else "cgroup v1 cpu.cfs_quota_us"
    if quota in {"max", "-1"}:
        return f"max ({source})"
    cores = _cores_from_quota(quota, period)
    if cores == "unknown":
        return "unknown"
    return f"{cores} cores ({source} {quota} {period})"


def parse_sample_line(line: str, offset):
    """One ``timestamp<TAB>nvidia-smi csv`` row. Unusable rows are skipped."""
    stripped = (line or "").strip()
    if not stripped or "\t" not in stripped:
        return None
    stamp, rest = stripped.split("\t", 1)
    stamp = stamp.strip()
    if not stamp.isdigit():
        return None
    remote = int(stamp)
    if isinstance(offset, (int, float)) and not isinstance(offset, bool):
        t_unix = remote + offset
    else:
        t_unix = remote
    try:
        cells = [cell.strip().replace("\r", "") for cell in next(csv.reader([rest]))]
    except csv.Error:
        return None
    gpu_index = 0  # Accept legacy single-GPU rows as GPU 0.
    if len(cells) == len(_SAMPLE_COLUMNS) + 1 and cells[0].isdigit():
        gpu_index = int(cells.pop(0))
    if len(cells) != len(_SAMPLE_COLUMNS):
        return None
    row = {
        "t_remote_unix": remote,
        "t_unix": round(float(t_unix), 3),
        "gpu_index": gpu_index,
    }
    row.update(zip(_SAMPLE_COLUMNS, cells, strict=True))
    return row


def _metric(text):
    if not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw or raw.upper() in {"N/A", "[N/A]", "UNKNOWN", "[NOT SUPPORTED]"}:
        return None
    token = raw.split()[0].replace("%", "")
    try:
        value = float(token)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _percentile(values, pct: float):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _round_meas(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, float)
        or not math.isfinite(value)
    ):
        return value
    rounded = round(value, 4)
    nearest = round(rounded)
    if abs(rounded - nearest) < 1e-9:
        return int(nearest)
    return rounded


def _low_gen_fraction(rows, phase_end, interval):
    busy = 0.0
    low = 0.0
    # Consecutive samples must belong to the same GPU, including multi-GPU SAM.
    following = {}
    for row in reversed(rows):
        gpu = row.get("gpu_index", 0)
        next_time = following.get(gpu, phase_end)
        following[gpu] = row["t_unix"]
        util = _metric(row.get("utilization_gpu", ""))
        weight = min(next_time - row["t_unix"], interval)
        if weight < 0 or util is None or util <= 0:
            continue
        current = _metric(row.get("pcie_gen_current", ""))
        maximum = _metric(row.get("pcie_gen_max", ""))
        if current is None or maximum is None:
            continue
        busy += weight
        if current < maximum:
            low += weight
    if busy <= 0:
        return "unknown"
    return _round_meas(low / busy)


def _phase_stats(rows, phase_end, interval):
    if not rows:
        return "unknown"

    def series(field):
        values = []
        for row in rows:
            number = _metric(row.get(field, ""))
            if number is not None:
                values.append(number)
        if not values:
            return "unknown"
        return {
            "min": _round_meas(min(values)),
            "median": _round_meas(_percentile(values, 50)),
            "p95": _round_meas(_percentile(values, 95)),
        }

    return {
        "samples": len(rows),
        "utilization_gpu": series("utilization_gpu"),
        "clock_sm_mhz": series("clock_sm"),
        "power_draw_w": series("power_draw"),
        "low_pcie_gen_while_busy": _low_gen_fraction(rows, phase_end, interval),
    }


def _phase_windows(eval_output, alias) -> dict:
    windows = {}
    if not eval_output:
        return windows
    base = Path(eval_output) / alias
    for split in ("validation", "test"):
        path = base / split / "summary.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        start, end = payload.get("started_at_unix"), payload.get("finished_at_unix")
        if isinstance(start, bool) or isinstance(end, bool):
            continue
        if (
            isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and math.isfinite(start)
            and math.isfinite(end)
        ):
            windows[split] = (float(start), float(end))
    return windows


def summarize_gpu_samples(samples, windows, interval):
    """Per-phase min/median/p95 and the share of busy time at a low PCIe gen."""
    usable = [row for row in samples if isinstance(row.get("t_unix"), (int, float))]
    if not usable:
        return "unknown"
    if not windows:
        times = [row["t_unix"] for row in usable]
        windows = {"run": (min(times), max(times) + interval)}
    out = {}
    for phase, bounds in windows.items():
        if not isinstance(bounds, tuple):
            out[phase] = "unknown"
            continue
        start, end = bounds
        chosen = [row for row in usable if start <= row["t_unix"] <= end]
        chosen.sort(key=lambda row: row["t_unix"])
        out[phase] = _phase_stats(chosen, end, interval)
        if chosen:
            out[phase]["gpus"] = {
                str(gpu): _phase_stats(
                    [row for row in chosen if row.get("gpu_index", 0) == gpu],
                    end,
                    interval,
                )
                for gpu in sorted({row.get("gpu_index", 0) for row in chosen})
            }
    return out


def _static_from_probe(probe) -> dict:
    unknown = {name: "unknown" for name in _STATIC_FIELDS}
    if not isinstance(probe, dict) or probe.get("status") != "ok":
        return unknown
    gpus = probe.get("gpus")
    host = probe.get("host") if isinstance(probe.get("host"), dict) else {}
    gpu = gpus[0] if isinstance(gpus, list) and gpus else {}
    driver = gpu.get("driver_version", "unknown")
    unknown["driver_version"] = driver if driver not in ("", None) else "unknown"
    for name in _STATIC_FIELDS:
        if name == "driver_version":
            continue
        value = host.get(name, "unknown")
        unknown[name] = value if value not in ("", None) else "unknown"
    return unknown


def _format_container_memory(raw: str) -> str:
    parts = raw.split()
    if len(parts) != 2 or parts[0] not in {"v1", "v2"}:
        return "unknown"
    kind, value = parts
    source = (
        "cgroup v2 memory.max" if kind == "v2" else "cgroup v1 memory.limit_in_bytes"
    )
    if value == "max":
        return f"max ({source})"
    if not value.isdigit():
        return "unknown"
    number = int(value)
    if kind == "v1" and number >= 1 << 62:
        return f"max ({source})"
    return f"{number} B ({source})"


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
            for name in (
                "accuracy",
                "final_unknown_questions",
                "final_missing_unique_units",
            )
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
    def __init__(self, launcher: str, model: str, engine: str, sample: bool = False):
        argv_model = _flag("--model")
        self.launcher = launcher
        self.model = argv_model or model
        self.engine = engine
        self.sample = bool(sample)
        self.eval_output = _flag("--eval-output")
        phase = _flag("--eval-phase")
        self.eval_phase = phase or "both"
        self.eval_phase_source = "argv" if phase else "launcher-default"
        self.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + os.urandom(4).hex()
        )
        self.root = logs_directory(self.model, self.eval_output)
        self.run_dir = self.root / self.run_id
        self.who = _who()
        self.git_commit = _git_commit()
        self.logging_errors = []
        self.offer = None
        resume = _flag("--resume-instance-id")
        self.instance_id = (
            int(resume) if resume and resume.isdigit() and int(resume) > 0 else None
        )
        self.release_status = "unknown"
        self._owns_eval_output = False
        self.probe = "unknown"
        self.score = "unknown"
        self.eval_returncode = "unknown"
        self._mono = {}
        self._timings = {
            name: "unknown"
            for name in (
                "rent_start_utc",
                "ready_utc",
                "eval_start_utc",
                "eval_end_utc",
                "release_utc",
            )
        }
        self._probed = False
        self._hooks_ready = False
        self._finished = False
        self._originals = {}
        self._sam_bound = False
        self._ssh_prefix = None
        self._expects_ready = False
        self._sampler_alive = False
        self._sample_seq = 0
        self._remote_pid = None
        self._remote_out = None
        self._samples = []
        self._static = None
        self._clock_offset = None
        self._sample_summary = "unknown"
        self.client_access = "unknown"
        self.metrics_diff = "unknown"

    def install(self) -> None:
        if self._sam_bound:
            return
        import tools.start_sam as sam

        self._originals = {
            "create_from_offers": sam.create_from_offers,
            "run": sam.run,
            "release_instance": sam.release_instance,
        }
        sam.create_from_offers = _wrap_create(sam.create_from_offers, self)
        sam.run = _wrap_run(sam.run, self)
        sam.release_instance = _wrap_release(sam.release_instance, self)
        self._sam_bound = True

    def note_create(self, offers, args, kwargs, created) -> None:
        contract = _contract(created)
        if contract is None:
            return
        if self.instance_id is not None and self.instance_id != contract:
            # SAM and detector may release a failed host and rent a replacement.
            # Keep each accepted rental's offer/probe/samples in its own entry.
            self.finish_once()
            hooks = {
                name: getattr(self, name)
                for name in (
                    "_sam_bound",
                    "_originals",
                    "_hooks_ready",
                    "_expects_ready",
                )
            }
            self.__init__(self.launcher, self.model, self.engine, self.sample)
            self.__dict__.update(hooks)
            self.mark_rent_start()
        self.instance_id = contract
        offer_id = kwargs.get("id", args[0] if args else None)
        selected = _find_offer(offers, offer_id)
        self.offer = selected
        self._write_offer()
        self._write_entry()
        self._update_manifest()
        _stderr_line(
            "run log: saved offer {}".format(_display(self.run_dir / "vast-offer.json"))
        )

    def mark_rent_start(self) -> None:
        if self._timings["rent_start_utc"] == "unknown":
            self._timings["rent_start_utc"] = _utc()
            self._mono["rent_start"] = time.monotonic()

    def note_ready(self) -> None:
        self._timings["ready_utc"] = _utc()
        self._mono["ready"] = time.monotonic()
        if self.sample:
            self.start_sampler()

    def note_eval_start(self) -> None:
        self._timings["eval_start_utc"] = _utc()
        self._mono["eval_start"] = time.monotonic()

    def note_eval_end(self, code) -> None:
        self._timings["eval_end_utc"] = _utc()
        self._mono["eval_end"] = time.monotonic()
        self.eval_returncode = code if isinstance(code, int) else "unknown"

    def note_setup(self, name: str, seconds) -> None:
        """Record weight download or env restore time, outside request latency."""
        if name not in {"weight_download_seconds", "env_restore_seconds"}:
            return
        setup = self._timings.get("setup")
        if not isinstance(setup, dict):
            setup = {}
            self._timings["setup"] = setup
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            setup[name] = "unknown"
        else:
            setup[name] = round(float(seconds), 3)
        self._write_entry()

    def note_release(self) -> None:
        self.release_status = "released"
        self._timings["release_utc"] = _utc()
        self._mono["release"] = time.monotonic()

    def arm_stage_hooks(self, caller_globals: dict) -> None:
        if self._hooks_ready:
            return
        self._hooks_ready = True
        fn = caller_globals.get("_wait_ready")
        if callable(fn) and not _mocked(fn) and not getattr(fn, "_gv_wrapped", False):
            caller_globals["_wait_ready"] = _wrap_ready(fn, self)
            self._expects_ready = True
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
        self._ssh_prefix = prefix
        try:
            result = subprocess.run(
                [*prefix, _PROBE],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.probe = "unknown"
            self._error("probe", exc)
            self._sample_without_ready_hook()
            return
        stdout = result.stdout or ""
        if result.returncode != 0:
            self.probe = {
                "status": "unknown",
                "gpus": "unknown",
                "host": "unknown",
                "raw": redact_text(stdout),
            }
            self._error("probe", f"nvidia-smi exited {result.returncode}")
            self._sample_without_ready_hook()
            return
        # Vast prints its SSH welcome banner on stderr. Keep stdout only.
        self.probe = parse_probe(stdout)
        self._write_entry()
        self._sample_without_ready_hook()

    def _sample_without_ready_hook(self) -> None:
        if self.sample and not self._expects_ready:
            self.start_sampler()

    def start_sampler(self) -> None:
        """Best-effort background nvidia-smi loop. Never raises."""
        if not self.sample or self._sampler_alive:
            return
        if not self._ssh_prefix:
            self._error("sample", "ssh prefix missing")
            return
        self._sample_seq += 1
        token = f"{self.run_id}-{self._sample_seq}"
        if not re.fullmatch(r"[A-Za-z0-9T-]+", token):
            self._error("sample", "unsafe sample path")
            return
        directory = f"/tmp/gv-gpu-{token}"
        script = directory + "/sample.sh"
        out = directory + "/samples.tsv"
        # Upload on the foreground SSH stdin. Background only the sampler;
        # otherwise POSIX sh gives cat /dev/null and reports a useless PID.
        # An exclusive private directory also prevents shared-/tmp symlink writes.
        launch = (
            f"umask 077; mkdir {directory} && cat > {script} && chmod 700 {script} && "
            f"{{ nohup sh {script} {out} {int(SAMPLE_INTERVAL)} "
            "</dev/null >/dev/null 2>&1 & echo $!; }"
        )
        try:
            if not isinstance(self.probe, dict) or self.probe.get("status") != "ok":
                probed = self._ssh(_PROBE, timeout=20)
                if probed.returncode == 0:
                    parsed = parse_probe(probed.stdout or "")
                    if parsed.get("status") == "ok":
                        self.probe = parsed
            self._static = _static_from_probe(self.probe)
            before = time.time()
            clock = self._ssh("date -u +%s", timeout=15)
            after = time.time()
            remote = (clock.stdout or "").strip()
            if clock.returncode == 0 and remote.isdigit():
                self._clock_offset = round(((before + after) / 2.0) - int(remote), 3)
            else:
                self._clock_offset = "unknown"
            started = self._ssh(launch, timeout=20, input=_SAMPLER_SH)
            lines = [
                line.strip()
                for line in (started.stdout or "").splitlines()
                if line.strip()
            ]
            pid = lines[-1] if lines else ""
            if started.returncode != 0 or not pid.isdigit():
                self._error("sample", "sampler did not start")
                return
            self._remote_pid = pid
            self._remote_out = out
            self._sampler_alive = True
        except Exception as exc:
            self._error("sample", exc)

    def stop_sampler(self) -> None:
        """Stop the remote loop and keep whatever samples were written."""
        if not self._sampler_alive:
            return
        pid, out = self._remote_pid, self._remote_out
        self._sampler_alive = False
        text = ""
        try:
            if pid and str(pid).isdigit() and out:
                result = self._ssh(f"kill {pid} 2>/dev/null; cat {out}", timeout=20)
                text = result.stdout or ""
                if result.returncode:
                    self._error("sample", "could not collect sampler output")
        except Exception as exc:
            self._error("sample", exc)
        try:
            self._ingest(text)
        except Exception as exc:
            self._error("sample", exc)

    def _ssh(self, remote: str, *, timeout: int, input: str | None = None):
        return subprocess.run(
            [*self._ssh_prefix, remote],
            input=input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )

    def _ingest(self, text: str) -> None:
        offset = (
            self._clock_offset if isinstance(self._clock_offset, (int, float)) else None
        )
        for line in (text or "").splitlines():
            row = parse_sample_line(line, offset)
            if row is not None:
                self._samples.append(row)
        if not self._samples:
            return
        _write_lines(self.run_dir / "gpu-samples.jsonl", self._samples)
        self._sample_summary = summarize_gpu_samples(
            self._samples,
            _phase_windows(self.eval_output, self.model)
            if self._owns_eval_output and offset is not None
            else {},
            SAMPLE_INTERVAL,
        )

    def capture_score(self) -> None:
        if not self._owns_eval_output:
            return
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
            self.stop_sampler()
        except Exception as exc:
            self._error("sample", exc)
        try:
            self._write_entry()
            self._update_manifest()
            self._write_entry()
            self._mirror_report()
            _stderr_line("run log: {}".format(_display(self.run_dir / "entry.json")))
            if self.logging_errors:
                _stderr_line(
                    f"run log: {len(self.logging_errors)} warning(s); "
                    "eval was not changed"
                )
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
            "instance_id": _measurement(self.instance_id)
            if self.instance_id is not None
            else "unknown",
            "who": self.who,
            "git_commit": self.git_commit,
            "timings": self._timing_block(),
            "hardware": {
                "offer": offer_fields(self.offer),
                "probe": self.probe,
                "samples": self._samples_block(),
                "client_access": self.client_access,
            },
            "metrics_diff": self.metrics_diff,
            "score": self.score,
            "eval_returncode": self.eval_returncode,
            "release_status": self.release_status,
            "logging_errors": list(self.logging_errors),
        }

    def _timing_block(self) -> dict:
        block = dict(self._timings)
        start = self._mono.get("rent_start")
        end = self._mono.get(
            "release", self._mono.get("eval_end", self._mono.get("finish"))
        )
        if start is None or end is None:
            block["wall_seconds"] = "unknown"
        else:
            block["wall_seconds"] = round(end - start, 3)
        return block

    def _row(self) -> dict:
        fields = offer_fields(self.offer)
        rented = self.instance_id is not None or self.offer is not None
        return {
            "run_id": self.run_id,
            "status": "rented" if rented else "not_rented",
            "time": self._timings["rent_start_utc"]
            if self._timings["rent_start_utc"] != "unknown"
            else _utc(),
            "who": self.who,
            "gpu": {
                "machine_id": fields["machine_id"],
                "gpu_name": fields["gpu_name"],
                "dph_total": fields["dph_total"],
                "geolocation": fields["geolocation"],
            },
            "score": short_score(self.score),
            "entry": self.run_id + "/entry.json",
            "samples": (
                self.run_id + "/gpu-samples.jsonl"
                if (self.run_dir / "gpu-samples.jsonl").is_file()
                else "unknown"
            ),
        }

    def _samples_block(self):
        if not self._samples:
            return "unknown"
        return {
            "path": self.run_id + "/gpu-samples.jsonl",
            "interval_seconds": SAMPLE_INTERVAL,
            "clock_offset_seconds": (
                self._clock_offset if self._clock_offset is not None else "unknown"
            ),
            "static": self._static or _static_from_probe("unknown"),
            "phases": self._sample_summary,
        }

    def _write_entry(self) -> None:
        _atomic_write(self.run_dir / "entry.json", self._entry())

    def _update_manifest(self) -> None:
        _upsert_manifest(self.root / "manifest.json", self._row(), self)

    def _mirror_report(self) -> None:
        if not self.eval_output or not self._owns_eval_output:
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
        samples = self.run_dir / "gpu-samples.jsonl"
        if samples.is_file():
            target = dest / samples.name
            target.write_text(
                samples.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
            )
        _upsert_manifest(report / "logs" / "manifest.json", self._row(), self)

    def _error(self, label, exc) -> None:
        message = "{} failed ({})".format(
            label, type(exc).__name__ if isinstance(exc, BaseException) else "error"
        )
        detail = redact_text(exc) if isinstance(exc, str) else ""
        if detail:
            message = f"{message}: {detail}"
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
        session.instance_id = instance_id
        try:
            session.stop_sampler()
        except Exception as exc:
            session._error("sample", exc)
        try:
            result = original(vast, instance_id)
        except BaseException:
            session.release_status = "failed"
            raise
        try:
            session.note_release()
        except Exception as exc:
            session._error("release", exc)
        return result

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
        output_was_absent = False
        try:
            output_was_absent = (
                bool(session.eval_output) and not Path(session.eval_output).exists()
            )
            session.note_eval_start()
        except Exception as exc:
            session._error("timer", exc)
        try:
            code = fn(*args, **kwargs)
        except BaseException:
            try:
                session.note_eval_end("unknown")
            except Exception as exc:
                session._error("timer", exc)
            raise
        session._owns_eval_output = output_was_absent
        try:
            session.note_eval_end(code)
        except Exception as exc:
            session._error("timer", exc)
        try:
            session.stop_sampler()
        except Exception as exc:
            session._error("sample", exc)
        try:
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
        out["p50_seconds"] = (
            _measurement(latency.get("p50_seconds"))
            if "p50_seconds" in latency
            else "unknown"
        )
        out["p95_seconds"] = (
            _measurement(latency.get("p95_seconds"))
            if "p95_seconds" in latency
            else "unknown"
        )
        out["latency_n"] = (
            _measurement(latency.get("n")) if "n" in latency else "unknown"
        )
        out["latency_source"] = (
            latency.get("source")
            if isinstance(latency.get("source"), str)
            else "unknown"
        )
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
        return (
            int(left) == int(right)
            and not isinstance(left, bool)
            and not isinstance(right, bool)
        )
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
        return "redacted" if _sanitize_text(value) != value else value
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
    return {
        "display": display,
        "operator_env": operator,
        "user": user,
        "git_user": git_user,
    }


def _git_user() -> str:
    return _git_line(["config", "--get", "user.name"])


def _git_commit() -> str:
    return _git_line(["rev-parse", "HEAD"])


def _git_line(args) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
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
            return item[len(prefix) :]
    return None


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dumps(value) -> str:
    return (
        json.dumps(redact(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(redact(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(tmp, path)


def _atomic_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    tmp = Path(filename)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_dumps(value))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
    if not acquired:
        session._error("manifest", "lock unavailable; index update skipped")
        return
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
            rows = [
                item
                for item in loaded
                if not (isinstance(item, dict) and item.get("run_id") == row["run_id"])
            ]
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
            # Never steal a lock from a paused writer. A crashed writer's lock
            # needs manual removal; the per-run entries remain intact.
            time.sleep(0.05)
        except PermissionError:
            # Windows can return access-denied when two exclusive creates race,
            # instead of file-exists. Wait and try again; do not delete the lock.
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
    _stderr_line(f"run log: {label} failed ({type(exc).__name__}); eval continues")


def _stderr_line(text: str) -> None:
    try:
        sys.stderr.write(redact_text(text) + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass
