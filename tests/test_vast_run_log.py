"""Offline tests for the Qwen Vast run log. No GPU and no Vast API."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from tools import start_sam
from tools import vast_run_log as run_log
from tools.vast_run_log import (
    _AUTO_ARM,
    _HOST_FIELDS,
    _PROBE,
    _SAMPLER_SH,
    LAUNCHERS,
    NATIVE_MODELS,
    SAMPLE_INTERVAL,
    RunLog,
    _wrap_create,
    _wrap_eval,
    _wrap_release,
    arm_if_launcher,
    attach,
    logs_directory,
    offer_fields,
    parse_probe,
    parse_sample_line,
    redact,
    summarize_gpu_samples,
    summarize_score,
)

REPO = Path(__file__).resolve().parents[1]


def _offer(**extra):
    body = {
        "id": 7,
        "machine_id": 4242,
        "host_id": 9,
        "geolocation": "California, US",
        "dph_total": 0.42,
        "gpu_name": "RTX_4090",
        "num_gpus": 1,
        "gpu_ram": 0,
        "gpu_frac": 0.5,
        "ssh_host": "ssh.example",
        "ssh_port": 2222,
        "api_key": "vast-secret-value",
        "ssh_key": "-----BEGIN PRIVATE KEY-----\nabc",
        "env": {"R2_SECRET_ACCESS_KEY": "r2-secret-value"},
        "description": "token sk-live-abcdefghij",
    }
    body.update(extra)
    return body


def test_redact_keeps_hardware_and_strips_secrets():
    cleaned = redact(_offer())
    assert cleaned["machine_id"] == 4242
    assert cleaned["ssh_host"] == "ssh.example"
    assert cleaned["gpu_ram"] == 0
    assert cleaned["api_key"] == "redacted"
    assert cleaned["ssh_key"] == "redacted"
    assert cleaned["env"] == "redacted"
    assert cleaned["description"] == "redacted"
    text = json.dumps(cleaned)
    assert "vast-secret-value" not in text
    assert "r2-secret-value" not in text
    assert "sk-live-abcdefghij" not in text


def test_missing_offer_fields_are_unknown_not_zero():
    fields = offer_fields({"id": 3, "gpu_ram": 0, "dph_total": float("nan")})
    assert fields["gpu_ram"] == 0
    assert fields["id"] == 3
    assert fields["dph_total"] == "unknown"
    assert fields["host_id"] == "unknown"
    assert fields["machine_id"] == "unknown"
    assert 0 not in {fields["host_id"], fields["machine_id"], fields["dph_total"]}


def test_probe_parser_and_bad_rows():
    text = (
        "RTX 4090, 0000:81:00.0, 4, 4, 8, 16, 450.00 W, 2520 MHz, 3105 MHz, "
        "10251 MHz, 550.90, 24564 MiB, 37 %\n"
        "HOST\tAMD EPYC 7763 64-Core Processor\t32\t257564 kB\t12.4\n"
    )
    parsed = parse_probe(text)
    assert parsed["status"] == "ok"
    assert parsed["gpus"][0]["pcie_gen_current"] == "4"
    assert parsed["gpus"][0]["pcie_width_max"] == "16"
    assert parsed["gpus"][0]["power_limit"] == "450.00 W"
    assert parsed["gpus"][0]["clock_sm"] == "2520 MHz"
    assert parsed["gpus"][0]["utilization_gpu"] == "37 %"
    assert parsed["gpus"][0]["driver_version"] == "550.90"
    assert parsed["host"]["cpu_model"] == "AMD EPYC 7763 64-Core Processor"
    assert parsed["host"]["host_cpu_cores"] == "32"
    assert parsed["host"]["host_ram_total"] == "257564 kB"
    assert parsed["host"]["cuda_version"] == "12.4"
    assert parsed["host"]["container_cpu_limit"] == "unknown"
    assert parsed["host"]["container_memory_limit"] == "unknown"
    assert "power-saving" in parsed["note"]
    assert "not throttling" in parsed["note"]
    short = (
        "RTX 4090, 0000:81:00.0, 4, 4, 8, 16, 450.00 W, 2520 MHz, 3105 MHz, "
        "10251 MHz, 550.90, 24564 MiB\n"
    )
    assert parse_probe(short)["status"] == "unknown"
    assert parse_probe("nope")["status"] == "unknown"
    assert parse_probe("")["gpus"] == "unknown"
    assert parse_probe("")["host"] == "unknown"


def test_probe_keeps_zero_utilization_and_cpu_commas_on_crlf():
    text = (
        "RTX 4090, 0000:81:00.0, 4, 4, 8, 16, 450.00 W, 2520 MHz, 3105 MHz, "
        "10251 MHz, 550.90, 24564 MiB, 0 %\r\n"
        "HOST\tIntel(R) Xeon(R) CPU, E5-2680 v4 @ 2.40GHz\t64\t0 kB\t12.8\r\n"
    )
    parsed = parse_probe(text)
    assert parsed["status"] == "ok"
    assert parsed["gpus"][0]["utilization_gpu"] == "0 %"
    assert parsed["gpus"][0]["pcie_gen_max"] == "4"
    assert parsed["gpus"][0]["pcie_width_current"] == "8"
    assert parsed["host"]["cpu_model"] == "Intel(R) Xeon(R) CPU, E5-2680 v4 @ 2.40GHz"
    assert parsed["host"]["host_cpu_cores"] == "64"
    assert parsed["host"]["host_ram_total"] == "0 kB"
    assert parsed["host"]["cuda_version"] == "12.8"


def test_probe_missing_host_facts_are_unknown_not_zero():
    gpus = (
        "RTX 4090, 0000:81:00.0, 4, 4, 8, 16, 450.00 W, 2520 MHz, 3105 MHz, "
        "10251 MHz, 550.90, 24564 MiB, 1 %\n"
    )
    missing = parse_probe(gpus)
    assert missing["status"] == "ok"
    assert missing["host"] == {name: "unknown" for name in _HOST_FIELDS}
    blank = parse_probe(gpus + "HOST\t\t\t\t\n")
    assert blank["host"] == {name: "unknown" for name in _HOST_FIELDS}
    assert "0" not in blank["host"].values()


def test_probe_command_asks_for_cpu_utilization_and_cuda():
    assert "utilization.gpu" in _PROBE
    assert "CUDA Version" in _PROBE
    assert "/proc/cpuinfo" in _PROBE
    assert "MemTotal" in _PROBE
    assert "_NPROCESSORS_ONLN" in _PROBE
    assert "/sys/fs/cgroup/cpu.max" in _PROBE
    assert "/sys/fs/cgroup/memory.max" in _PROBE
    assert "cpu.cfs_quota_us" in _PROBE
    assert "memory.limit_in_bytes" in _PROBE
    assert "[[" not in _PROBE
    assert "wmic" not in _PROBE
    assert "\r" not in _PROBE
    assert _PROBE.count('exit "$smi_status"') == 1


def _live_probe_streams():
    text = (REPO / "tests/fixtures/vast_probe_raw.txt").read_text(encoding="utf-8")
    stdout, stderr = text.split("[stderr]\n", 1)
    return stdout, stderr


def test_live_probe_banner_does_not_drop_the_gpu_row():
    """Regression for instance 60852547: Vast's SSH banner arrives on stderr."""
    stdout, stderr = _live_probe_streams()
    combined = stdout + stderr
    parsed = parse_probe(combined)
    gpu = parsed["gpus"][0]
    assert parsed["status"] == "ok"
    assert gpu["name"] == "NVIDIA GeForce RTX 4090"
    assert gpu["pci_bus_id"] == "00000000:C1:00.0"
    assert gpu["pcie_gen_current"] == "1"
    assert gpu["pcie_gen_max"] == "4"
    assert gpu["pcie_width_current"] == "16"
    assert gpu["pcie_width_max"] == "16"
    assert gpu["power_limit"] == "450.00 W"
    assert gpu["clock_sm"] == "210 MHz"
    assert gpu["clock_max_sm"] == "3150 MHz"
    assert gpu["utilization_gpu"] == "0 %"
    assert gpu["driver_version"] == "595.99.02"
    assert parsed["host"]["cpu_model"] == "AMD EPYC 7702P 64-Core Processor"
    assert parsed["host"]["host_cpu_cores"] == "128"
    assert parsed["host"]["host_ram_total"] == "527980792 kB"
    assert parsed["host"]["cuda_version"] == "13.2"
    assert parsed["host"]["container_cpu_limit"] == "unknown"
    assert parsed["host"]["container_memory_limit"] == "unknown"
    assert "Welcome to vast.ai" in combined
    assert "power-saving" in parsed["note"]
    assert "Have fun!" not in json.dumps(parsed["gpus"])
    offer = json.loads(
        (REPO / "tests/fixtures/vast_offer.json").read_text(encoding="utf-8")
    )
    fields = offer_fields(offer)
    assert fields["verification"] == "verified"
    assert "verified" not in fields
    assert fields["cpu_cores_effective"] == 32.0
    assert fields["cpu_ram"] == 128901
    assert fields["gpu_frac"] == 0.25
    assert fields["machine_id"] == 12592
    assert "84.1.117.74" not in json.dumps(fields)


def test_container_limits_are_labeled_apart_from_the_host(tmp_path, monkeypatch):
    stdout = (
        "NVIDIA GeForce RTX 4090, 00000000:C1:00.0, 1, 4, 16, 16, 450.00 W, "
        "210 MHz, 3150 MHz, 405 MHz, 595.99.02, 24564 MiB, 0 %\n"
        "HOST\tAMD EPYC 7702P 64-Core Processor\t128\t527980792 kB\t13.2\t"
        "v2 3200000 100000\tv2 135107444736\n"
    )
    parsed = parse_probe(stdout + "Welcome to vast.ai. Have fun!\n")
    assert parsed["status"] == "ok"
    assert parsed["host"]["host_cpu_cores"] == "128"
    assert parsed["host"]["host_ram_total"] == "527980792 kB"
    assert parsed["host"]["container_cpu_limit"] == (
        "32 cores (cgroup v2 cpu.max 3200000 100000)"
    )
    assert parsed["host"]["container_memory_limit"] == (
        "135107444736 B (cgroup v2 memory.max)"
    )
    v1 = parse_probe(
        "RTX 4090, 0000:81:00.0, 4, 4, 8, 16, 450.00 W, 2520 MHz, 3105 MHz, "
        "10251 MHz, 550.90, 24564 MiB, 0 %\n"
        "HOST\tAMD EPYC\t128\t527980792 kB\t13.2\t"
        "v1 -1 100000\tv1 9223372036854771712\n"
    )
    assert v1["host"]["container_cpu_limit"] == "max (cgroup v1 cpu.cfs_quota_us)"
    assert (
        v1["host"]["container_memory_limit"] == "max (cgroup v1 memory.limit_in_bytes)"
    )
    partial = parse_probe(
        "RTX 4090, 0000:81:00.0, 4, 4, 8, 16, 450.00 W, 2520 MHz, 3105 MHz, "
        "10251 MHz, 550.90, 24564 MiB, 0 %\n"
        "HOST\tAMD EPYC\t128\t527980792 kB\t13.2\tv1 150000 100000\tv2 0\n"
    )
    assert partial["host"]["container_cpu_limit"] == (
        "1.5 cores (cgroup v1 cpu.cfs_quota_us 150000 100000)"
    )
    assert partial["host"]["container_memory_limit"] == "0 B (cgroup v2 memory.max)"
    zero = parse_probe(
        "RTX 4090, 0000:81:00.0, 4, 4, 8, 16, 450.00 W, 2520 MHz, 3105 MHz, "
        "10251 MHz, 550.90, 24564 MiB, 0 %\n"
        "HOST\tAMD EPYC\t128\t527980792 kB\t13.2\tv2 0 100000\tv2 max\n"
    )
    assert zero["host"]["container_cpu_limit"].startswith("0 cores ")
    assert zero["host"]["container_memory_limit"] == "max (cgroup v2 memory.max)"

    banner = (
        "Welcome to vast.ai. If authentication fails, try again after a few seconds, "
        "and double check your ssh key.\nHave fun!\n"
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout, banner)

    monkeypatch.setattr("tools.vast_run_log.subprocess.run", fake_run)
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    session = RunLog("tools.start_qwen8b_vllm", "qwen8b_r2_vllm", "vllm")
    session.maybe_probe(["ssh", "-p", "22", "root@example", "true"])
    assert session.probe["status"] == "ok"
    assert session.probe["gpus"][0]["clock_sm"] == "210 MHz"
    assert "raw" not in session.probe
    saved = json.loads((session.run_dir / "entry.json").read_text(encoding="utf-8"))
    assert saved["hardware"]["probe"]["host"]["container_cpu_limit"].startswith(
        "32 cores"
    )
    assert "Welcome to vast.ai" not in json.dumps(saved["hardware"]["probe"])
    assert "Have fun!" not in json.dumps(saved["hardware"]["probe"])


def test_probe_raw_keeps_stdout_and_drops_the_ssh_banner(tmp_path, monkeypatch):
    stdout = "nvidia-smi failed before the table\n"
    banner = "Welcome to vast.ai. Have fun!\n"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout, banner)

    monkeypatch.setattr("tools.vast_run_log.subprocess.run", fake_run)
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    session = RunLog("tools.start_qwen8b_vllm", "qwen8b_r2_vllm", "vllm", True)
    session.maybe_probe(["ssh", "-p", "22", "root@example", "true"])
    assert session.probe["status"] == "unknown"
    assert session.probe["raw"] == stdout
    assert "Welcome to vast.ai" not in session.probe["raw"]
    assert "Have fun!" not in json.dumps(session.probe)
    session.finish_once()
    saved = json.loads((session.run_dir / "entry.json").read_text(encoding="utf-8"))
    assert saved["hardware"]["probe"]["raw"] == stdout
    assert "Welcome to vast.ai" not in saved["hardware"]["probe"]["raw"]


def test_finish_without_a_rental_is_not_rented(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    missed = RunLog("tools.start_qwen8b_vllm", "qwen8b_r2_vllm", "vllm", True)
    missed.finish_once()
    rows = json.loads((missed.root / "manifest.json").read_text(encoding="utf-8"))
    assert rows[0]["status"] == "not_rented"
    assert rows[0]["gpu"]["machine_id"] == "unknown"
    assert rows[0]["run_id"] == missed.run_id
    resumed = RunLog("tools.start_qwen8b_vllm", "qwen8b_r2_vllm", "vllm", True)
    resumed.instance_id = 60852547
    resumed.finish_once()
    rows = json.loads((missed.root / "manifest.json").read_text(encoding="utf-8"))
    by_id = {row["run_id"]: row for row in rows}
    assert by_id[missed.run_id]["status"] == "not_rented"
    assert by_id[resumed.run_id]["status"] == "rented"
    assert by_id[resumed.run_id]["gpu"]["machine_id"] == "unknown"


def test_score_summary_preserves_zero_and_marks_missing_unknown(tmp_path):
    alias = "qwen8b_r2_host"
    split = tmp_path / alias / "validation"
    split.mkdir(parents=True)
    (split / "summary.json").write_text(
        json.dumps(
            {
                "accuracy": 0,
                "correct": 0,
                "planned_questions": 24,
                "final_unknown_questions": 2,
                "final_missing_unique_units": 0,
                "test_attempt_latency": {"n": 3, "source": "transport"},
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "run-manifest.json").write_text(
        json.dumps({"slots_per_model": 12}), encoding="utf-8", newline="\n"
    )
    score = summarize_score(tmp_path, alias)
    assert score["validation"]["accuracy"] == 0
    assert score["validation"]["final_unknown_questions"] == 2
    assert score["validation"]["p50_seconds"] == "unknown"
    assert score["validation"]["p95_seconds"] == "unknown"
    assert score["test"] == "unknown"
    assert score["concurrency_slots"] == 12
    assert summarize_score(tmp_path / "missing", alias) == "unknown"


def test_selected_offer_is_saved_without_changing_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("GOODVISION_OPERATOR", "colleague")
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.setenv("USERNAME", "win-user")
    cheap = _offer(id=1, machine_id=10, dph_total=0.2, geolocation="A")
    chosen = _offer(
        id=2, machine_id=20, dph_total=0.4, geolocation="B", gpu_name="RTX_4090"
    )
    calls = []

    class Vast:
        def create_instance(self, **kwargs):
            calls.append(kwargs["id"])
            if kwargs["id"] == 1:
                return {"success": False, "msg": "offer 1 is not available"}
            return {"success": True, "new_contract": 99}

    session = RunLog("tools.start_qwen8b_r2", "qwen8b_r2_host", "transformers")
    session.eval_output = None
    wrapped = _wrap_create(start_sam.create_from_offers, session)
    vast = Vast()
    created = wrapped(vast, [cheap, chosen], disk_gb=80, label="goodvision-qwen8b-test")
    assert created == 99
    assert calls == [1, 2]
    saved = json.loads(
        (session.run_dir / "vast-offer.json").read_text(encoding="utf-8")
    )
    raw = (session.run_dir / "vast-offer.json").read_bytes()
    assert b"\r" not in raw
    assert saved["machine_id"] == 20
    assert saved["geolocation"] == "B"
    assert saved["api_key"] == "redacted"
    assert b"vast-secret-value" not in raw
    assert b"r2-secret-value" not in raw
    entry = json.loads((session.run_dir / "entry.json").read_text(encoding="utf-8"))
    assert entry["hardware"]["offer"]["machine_id"] == 20
    assert entry["hardware"]["offer"]["host_id"] == 9
    assert entry["who"]["display"] == "colleague"
    assert entry["who"]["user"] == "win-user"
    assert entry["instance_id"] == 99
    rows = json.loads((session.root / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["gpu"] == {
        "machine_id": 20,
        "gpu_name": "RTX_4090",
        "dph_total": 0.4,
        "geolocation": "B",
    }
    assert rows[0]["who"]["operator_env"] == "colleague"
    assert rows[0]["status"] == "rented"
    assert rows[0]["score"] == "unknown"
    report = tmp_path / "report"
    (report / "qwen8b_r2_host" / "validation").mkdir(parents=True)
    (report / "qwen8b_r2_host" / "validation" / "summary.json").write_text(
        json.dumps(
            {
                "accuracy": 0,
                "final_unknown_questions": 1,
                "final_missing_unique_units": 0,
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    session.eval_output = str(report)
    session._owns_eval_output = True
    session.capture_score()
    session._finished = False
    session.finish_once()
    rows = json.loads((session.root / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["score"]["validation"]["accuracy"] == 0
    assert rows[0]["score"]["test"] == "unknown"


def test_second_run_appends_a_row_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    first = RunLog("tools.start_qwen8b_vllm", "qwen8b_r2_vllm", "vllm")
    second = RunLog("tools.start_qwen8b_vllm", "qwen8b_r2_vllm", "vllm")
    first.note_create(
        [_offer(id=5, machine_id=50)],
        (),
        {"id": 5},
        {"success": True, "new_contract": 5},
    )
    second.note_create(
        [_offer(id=6, machine_id=60, geolocation="EU")],
        (),
        {"id": 6},
        {"success": True, "new_contract": 6},
    )
    rows = json.loads((first.root / "manifest.json").read_text(encoding="utf-8"))
    assert [row["gpu"]["machine_id"] for row in rows] == [50, 60]
    assert (
        json.loads((first.run_dir / "vast-offer.json").read_text(encoding="utf-8"))[
            "machine_id"
        ]
        == 50
    )
    assert (second.run_dir / "vast-offer.json").is_file()


def test_log_write_failure_does_not_change_the_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    session = RunLog("tools.start_qwen8b_r2", "qwen8b_r2_host", "transformers")
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x", encoding="utf-8")
    session.root = blocked
    session.run_dir = blocked / session.run_id

    class Vast:
        def create_instance(self, **kwargs):
            return {"success": True, "new_contract": 77}

    wrapped = _wrap_create(start_sam.create_from_offers, session)
    assert wrapped(Vast(), [_offer()], disk_gb=80, label="label") == 77
    assert session.logging_errors
    assert not (tmp_path / "out").exists()


def test_eval_wrapper_returns_the_original_result_when_logging_breaks():
    session = RunLog("tools.start_qwen8b_r2", "qwen8b_r2_host", "transformers")

    def explode():
        raise RuntimeError("disk full")

    session.capture_score = explode
    wrapped = _wrap_eval(lambda: 0, session)
    assert wrapped() == 0
    assert session.eval_returncode == 0
    assert session.logging_errors


def test_score_lands_in_the_model_report_without_creating_a_missing_output(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "series"))
    report_root = tmp_path / "eval-out"
    alias = "qwen8b_r2_host"
    (report_root / alias / "test").mkdir(parents=True)
    (report_root / alias / "test" / "summary.json").write_text(
        json.dumps(
            {
                "accuracy": 0.25,
                "final_unknown_questions": 4,
                "final_missing_unique_units": 1,
                "test_attempt_latency": {
                    "p50_seconds": 1.5,
                    "p95_seconds": 3.5,
                    "n": 10,
                    "source": "transport",
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    session = RunLog("tools.start_qwen8b_r2", alias, "transformers")
    session.eval_output = str(report_root)
    session._owns_eval_output = True
    session.capture_score()
    session.mark_rent_start()
    session.note_create(
        [_offer()], (), {"id": 7}, {"success": True, "new_contract": 11}
    )
    session.finish_once()
    copied = report_root / alias / "logs" / session.run_id / "entry.json"
    entry = json.loads(copied.read_text(encoding="utf-8"))
    assert entry["score"]["test"]["accuracy"] == 0.25
    assert entry["score"]["test"]["p50_seconds"] == 1.5
    assert entry["timings"]["rent_start_utc"] != "unknown"
    assert isinstance(entry["timings"]["wall_seconds"], float)
    mirror = json.loads(
        (report_root / alias / "logs" / "manifest.json").read_text(encoding="utf-8")
    )
    assert mirror[0]["score"]["test"]["final_missing_unique_units"] == 1
    absent = tmp_path / "brand-new-output"
    assert logs_directory(alias, absent) != absent
    assert not absent.exists()


def test_log_root_is_not_the_eval_output(tmp_path, monkeypatch):
    eval_output = tmp_path / "outputs" / "qwen8b_r2_host"
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(eval_output))
    root = logs_directory("qwen8b_r2_host", eval_output)
    root.mkdir(parents=True)
    assert not eval_output.exists()
    assert root.resolve() != eval_output.resolve()
    assert not root.resolve().is_relative_to(eval_output.resolve())


@pytest.mark.parametrize("module", ["tools.start_qwen8b_r2", "tools.start_qwen8b_vllm"])
def test_launcher_module_writes_a_log_and_still_rejects_a_bad_bundle(tmp_path, module):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["GOODVISION_RUN_LOG_ROOT"] = str(tmp_path / "logs")
    env["GOODVISION_OPERATOR"] = "colleague"
    env["VAST_API_KEY"] = "vast-secret-should-not-be-logged"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    output = tmp_path / "eval-out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--eval-bundle",
            str(bundle),
            "--eval-output",
            str(output),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert not output.exists()
    combined = result.stdout + result.stderr
    assert "vast-secret-should-not-be-logged" not in combined
    logs = list((tmp_path / "logs").rglob("entry.json"))
    assert len(logs) == 1
    entry = json.loads(logs[0].read_text(encoding="utf-8"))
    assert entry["who"]["display"] == "colleague"
    assert entry["hardware"]["offer"]["machine_id"] == "unknown"
    assert b"\r" not in logs[0].read_bytes()
    text = logs[0].read_text(encoding="utf-8")
    assert "vast-secret-should-not-be-logged" not in text
    assert "run log:" in result.stderr
    manifest = json.loads(next((tmp_path / "logs").rglob("manifest.json")).read_text())
    assert manifest[0]["status"] == "not_rented"
    assert entry["instance_id"] == "unknown"


def test_every_vast_rental_calls_the_recorder():
    needle = "from vastai import VastAI"
    found = []
    for path in (REPO / "tools").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if needle not in text:
            continue
        rel = path.relative_to(REPO).as_posix()
        found.append(rel)
        if path.name in _AUTO_ARM:
            assert "vast_run_log import attach" not in text
            continue
        assert "vast_run_log import attach" in text
        assert text.count("\n    attach()") >= 1 or "\n    attach()\n" in text
        if path.name != "start_sam.py":
            assert ".create_instance(" not in text
    direct = [
        path.relative_to(REPO).as_posix()
        for path in (REPO / "tools").rglob("*.py")
        if ".create_instance(" in path.read_text(encoding="utf-8")
    ]
    assert direct == ["tools/start_sam.py"]
    helper = (REPO / "tools/start_sam.py").read_text(encoding="utf-8")
    body = helper.split("def create_from_offers(", 1)[1].split("\ndef ", 1)[0]
    assert "vast.create_instance(" in body
    outside = helper.replace(body, "", 1)
    assert ".create_instance(" not in outside
    assert set(found) == {
        "tools/diag_qwen8b_vllm_import.py",
        "tools/r2_pull_bench.py",
        "tools/start_detector.py",
        "tools/start_qwen8b_r2.py",
        "tools/start_qwen8b_vllm.py",
        "tools/start_sam.py",
        "tools/start_bag_demo.py",
    }
    init = (REPO / "tools/__init__.py").read_text(encoding="utf-8")
    assert "arm_if_launcher" in init
    assert _AUTO_ARM == {"start_qwen8b_r2.py"}
    assert LAUNCHERS["start_qwen8b_r2.py"][3] is False
    for name in (
        "start_qwen8b_vllm.py",
        "start_sam.py",
        "start_bag_demo.py",
        "start_detector.py",
        "diag_qwen8b_vllm_import.py",
        "r2_pull_bench.py",
    ):
        assert LAUNCHERS[name][3] is True
    assert "pip " not in _SAMPLER_SH
    assert "nvidia-smi --query-gpu=" in _SAMPLER_SH
    assert 'sleep "$interval"' in _SAMPLER_SH
    assert SAMPLE_INTERVAL == 2
    skill = (REPO / ".cursor/skills/vast-run-log/SKILL.md").read_text(encoding="utf-8")
    assert "attach()" in skill
    assert "outputs/run-logs/<alias>/logs/manifest.json" in skill
    assert "<eval-output>/<alias>/logs/" in skill
    assert "git user.name" in skill
    assert "python -m tools.start_qwen8b_vllm" in skill
    assert "create_from_offers" in skill
    assert "release_instance" in skill
    assert "GOODVISION_RUN_LOG_ROOT" in skill
    assert "is not recorded" in skill
    assert "did not exist" in skill


def test_gpu_sample_summary_keeps_zero_and_low_pcie_fraction():
    rows = []
    samples = (
        (100, "1, 16, 4, 16, 20.00 W, 450.00 W, 210 MHz, 0 %, 10 MiB, 35"),
        (102, "1, 16, 4, 16, 100.00 W, 450.00 W, 1000 MHz, 10 %, 1000 MiB, 40"),
        (104, "4, 16, 4, 16, 200.00 W, 450.00 W, 2500 MHz, 80 %, 8000 MiB, 55"),
        (106, "4, 16, 4, 16, 300.00 W, 450.00 W, 2700 MHz, 90 %, 9000 MiB, 60"),
        (108, "1, 16, 4, 16, 400.00 W, 450.00 W, 3000 MHz, 100 %, 12000 MiB, 70"),
    )
    for stamp, csv in samples:
        rows.append(parse_sample_line(f"{stamp}\t{csv}\r\n", 0))
    assert rows[0]["utilization_gpu"] == "0 %"
    assert parse_sample_line("Welcome to vast.ai. Have fun!\n", 0) is None
    assert (
        parse_sample_line(
            "100\tN/A, 16, 4, 16, [N/A], 450.00 W, 210 MHz, N/A, 10 MiB, 35\n", 0
        )["power_draw"]
        == "[N/A]"
    )
    summary = summarize_gpu_samples(rows, {"validation": (100, 109)}, SAMPLE_INTERVAL)
    stats = summary["validation"]
    assert stats["utilization_gpu"] == {"min": 0, "median": 80, "p95": 98}
    assert stats["clock_sm_mhz"]["min"] == 210
    assert stats["clock_sm_mhz"]["median"] == 2500
    assert stats["power_draw_w"]["median"] == 200
    assert stats["low_pcie_gen_while_busy"] == 0.4286
    assert summarize_gpu_samples([], {}, SAMPLE_INTERVAL) == "unknown"


def test_sampler_stops_before_release_and_records_the_phase(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setattr("tools.vast_run_log.time.time", lambda: 100.0)
    tsv = (
        "100\t1, 16, 4, 16, 20.00 W, 450.00 W, 210 MHz, 0 %, 10 MiB, 35\n"
        "102\t1, 16, 4, 16, 100.00 W, 450.00 W, 1000 MHz, 10 %, 1000 MiB, 40\n"
        "104\t4, 16, 4, 16, 200.00 W, 450.00 W, 2500 MHz, 80 %, 8000 MiB, 55\n"
        "106\t4, 16, 4, 16, 300.00 W, 450.00 W, 2700 MHz, 90 %, 9000 MiB, 60\n"
        "108\t1, 16, 4, 16, 400.00 W, 450.00 W, 3000 MHz, 100 %, 12000 MiB, 70\n"
        "Welcome to vast.ai. Have fun!\n"
    )

    def fake_run(command, **kwargs):
        remote = command[-1]
        if remote == "date -u +%s":
            return subprocess.CompletedProcess(command, 0, "100\n", "")
        if remote.startswith("umask 077;"):
            return subprocess.CompletedProcess(
                command, 0, "4242\n", "Welcome to vast.ai. Have fun!\n"
            )
        if remote.startswith("kill "):
            return subprocess.CompletedProcess(command, 0, tsv, "")
        return subprocess.CompletedProcess(command, 1, "", "missing")

    monkeypatch.setattr("tools.vast_run_log.subprocess.run", fake_run)
    session = RunLog("tools.start_qwen8b_vllm", "qwen8b_r2_vllm", "vllm", True)
    session._ssh_prefix = ["ssh", "-p", "22", "root@example"]
    session._expects_ready = True
    session.probe = parse_probe(
        "NVIDIA GeForce RTX 4090, 00000000:C1:00.0, 1, 4, 16, 16, 450.00 W, "
        "210 MHz, 3150 MHz, 405 MHz, 595.99.02, 24564 MiB, 0 %\n"
        "HOST\tAMD EPYC 7702P 64-Core Processor\t128\t527980792 kB\t13.2\n"
    )
    phase = tmp_path / "eval" / "qwen8b_r2_vllm" / "validation"
    phase.mkdir(parents=True)
    (phase / "summary.json").write_text(
        json.dumps(
            {
                "accuracy": 0,
                "started_at_unix": 100,
                "finished_at_unix": 109,
            }
        ),
        encoding="utf-8",
    )
    session.eval_output = str(tmp_path / "eval")
    session._owns_eval_output = True
    order = []

    def release(vast, instance_id):
        order.append(instance_id)

    session.note_ready()
    assert session._sampler_alive
    _wrap_release(release, session)(None, 7)
    assert order == [7]
    assert session._sampler_alive is False
    session.finish_once()
    entry = json.loads((session.run_dir / "entry.json").read_text(encoding="utf-8"))
    block = entry["hardware"]["samples"]
    assert block["interval_seconds"] == 2
    assert block["static"]["driver_version"] == "595.99.02"
    assert block["static"]["cuda_version"] == "13.2"
    assert block["static"]["host_cpu_cores"] == "128"
    assert block["phases"]["validation"]["utilization_gpu"]["min"] == 0
    assert block["phases"]["validation"]["low_pcie_gen_while_busy"] == 0.4286
    body = (session.run_dir / "gpu-samples.jsonl").read_text(encoding="utf-8")
    assert "Welcome to vast.ai" not in body
    assert "\r" not in body
    manifest = json.loads((session.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["samples"].endswith("/gpu-samples.jsonl")


def test_hf_log_does_not_start_gpu_sampling(monkeypatch):
    session = RunLog(*LAUNCHERS["start_qwen8b_r2.py"])
    session._ssh_prefix = ["ssh", "root@example"]
    calls = []
    monkeypatch.setattr(
        "tools.vast_run_log.subprocess.run",
        lambda *args, **kwargs: (
            calls.append(args) or subprocess.CompletedProcess(args, 0, "", "")
        ),
    )
    session.note_ready()
    assert session.sample is False
    assert calls == []


def test_skill_example_records_helper_calls(tmp_path, monkeypatch):
    """Execute the documented integration with fake rental/SSH helpers."""
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setattr(run_log, "_SESSION", None)
    monkeypatch.setattr(run_log.atexit, "register", lambda fn: None)
    monkeypatch.setattr(sys, "argv", ["example_vast_run.py"])
    monkeypatch.setitem(
        LAUNCHERS, "example_vast_run.py", ("example_vast_run", "example", "fake", True)
    )
    calls = []

    class FakeVast:
        def create_instance(self, offer_id, **kwargs):
            calls.append(("create", offer_id))
            return {"success": True, "new_contract": 101}

    def create(vast, offers, **kwargs):
        return vast.create_instance(offers[0]["id"], **kwargs)["new_contract"]

    def run(command, **kwargs):
        calls.append(("ssh", command))
        return subprocess.CompletedProcess(command, 0, "", "")

    def release(vast, instance_id):
        calls.append(("release", instance_id))

    monkeypatch.setattr(start_sam, "create_from_offers", create)
    monkeypatch.setattr(start_sam, "run", run)
    monkeypatch.setattr(start_sam, "release_instance", release)
    monkeypatch.setattr(
        RunLog, "maybe_probe", lambda self, command: calls.append(("probe", command))
    )
    monkeypatch.setattr(RunLog, "arm_stage_hooks", lambda self, namespace: None)
    skill = (REPO / ".cursor/skills/vast-run-log/SKILL.md").read_text()
    example = skill.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {
        "__file__": "example_vast_run.py",
        "vast": FakeVast(),
        "offers": [_offer()],
        "disk_gb": 20,
        "label": "offline-example",
        "ssh": ["ssh", "root@example"],
    }
    exec(compile(example, namespace["__file__"], "exec"), namespace)
    namespace["main"]()
    session = run_log._SESSION
    assert session.instance_id == 101
    assert session.offer["machine_id"] == 4242
    assert session.release_status == "released"
    assert [name for name, _ in calls] == ["create", "ssh", "probe", "release"]
    session.finish_once()
    row = json.loads((session.root / "manifest.json").read_text())[0]
    assert row["status"] == "rented"
    assert row["gpu"]["machine_id"] == 4242


def test_attach_patches_detector_ready_hook(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    import tools.vast_run_log as mod

    monkeypatch.setattr(mod, "_SESSION", None)
    ready = []

    def wait_for_worker(*args, **kwargs):
        ready.append("worker")

    namespace = {
        "__file__": str(REPO / "tools" / "start_detector.py"),
        "wait_for_worker": wait_for_worker,
        "run": lambda *args, **kwargs: None,
        "release_instance": lambda *args, **kwargs: None,
        "create_from_offers": lambda *args, **kwargs: 1,
    }
    attach(namespace)
    assert namespace["wait_for_worker"].__dict__.get("_gv_wrapped") or getattr(
        namespace["wait_for_worker"], "_gv_wrapped", False
    )
    namespace["wait_for_worker"]()
    assert ready == ["worker"]
    assert mod._SESSION.sample is True
    assert mod._SESSION._expects_ready is True


@pytest.mark.parametrize("module", ["tools.start_qwen8b_r2", "tools.start_qwen8b_vllm"])
def test_rejected_launch_does_not_adopt_an_existing_report(tmp_path, module):
    alias = "qwen8b_r2_vllm" if module.endswith("vllm") else "qwen8b_r2_host"
    output = tmp_path / "previous-run"
    split = output / alias / "test"
    split.mkdir(parents=True)
    (split / "summary.json").write_text('{"accuracy": 1.0}')
    env = dict(os.environ, GOODVISION_RUN_LOG_ROOT=str(tmp_path / "logs"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--eval-bundle",
            str(tmp_path / "missing"),
            "--eval-output",
            str(output),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    entry = json.loads(next((tmp_path / "logs").rglob("entry.json")).read_text())
    assert entry["score"] == "unknown"
    assert not (output / alias / "logs").exists()


def test_eval_owns_only_its_fresh_output(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    session = RunLog("tools.start_qwen8b_r2", "qwen8b_r2_host", "transformers")
    output = tmp_path / "eval"
    session.eval_output = str(output)

    def evaluate():
        split = output / session.model / "test"
        split.mkdir(parents=True)
        (split / "summary.json").write_text('{"accuracy": 0}')
        return 1

    assert _wrap_eval(evaluate, session)() == 1
    assert session.score["test"]["accuracy"] == 0
    # The score is captured at this invocation's return, never re-read at exit.
    (output / session.model / "test" / "summary.json").write_text('{"accuracy": 1}')
    session.finish_once()
    assert session.score["test"]["accuracy"] == 0
    other = RunLog("tools.start_qwen8b_r2", session.model, "transformers")
    other.eval_output = str(output)
    assert _wrap_eval(lambda: 1, other)() == 1
    other.finish_once()
    assert other.score == "unknown"


@pytest.mark.parametrize("error", [FileExistsError, KeyboardInterrupt])
def test_failed_eval_cannot_claim_a_concurrent_report(tmp_path, monkeypatch, error):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    session = RunLog("tools.start_qwen8b_r2", "qwen8b_r2_host", "transformers")
    session.eval_output = str(tmp_path / "eval")

    def race():
        split = Path(session.eval_output) / session.model / "test"
        split.mkdir(parents=True)
        (split / "summary.json").write_text('{"accuracy": 1}')
        raise error()

    with pytest.raises(error):
        _wrap_eval(race, session)()
    session.finish_once()
    assert session.score == "unknown"
    assert not (Path(session.eval_output) / session.model / "logs").exists()


def test_manifest_lock_timeout_preserves_the_previous_index(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path))
    session = RunLog("tools.start_qwen8b_r2", "qwen8b_r2_host", "transformers")
    session.root.mkdir(parents=True)
    path = session.root / "manifest.json"
    old = '[{"run_id":"existing"}]'
    path.write_text(old)
    lock = session.root / ".manifest.lock"
    lock.write_text("")
    os.utime(lock, (1, 1))
    monkeypatch.setattr(run_log.time, "sleep", lambda _: None)
    session._update_manifest()
    assert path.read_text() == old
    assert lock.exists()
    assert session.logging_errors


def test_concurrent_manifest_writers_preserve_every_run(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    path = tmp_path / "manifest.json"
    errors = []
    session = SimpleNamespace(_error=lambda *args: errors.append(args))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda i: run_log._upsert_manifest(path, {"run_id": str(i)}, session),
                range(24),
            )
        )
    assert not errors
    assert {row["run_id"] for row in json.loads(path.read_text())} == {
        str(i) for i in range(24)
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_release_is_not_recorded_as_released(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path))
    session = RunLog("tools.start_qwen8b_r2", "qwen8b_r2_host", "transformers")
    session.mark_rent_start()
    vast = SimpleNamespace(
        client=SimpleNamespace(timeout=10),
        destroy_instance=lambda **kwargs: {"success": False},
    )
    release = run_log._wrap_release(start_sam.release_instance, session)
    with pytest.raises(RuntimeError, match="did not confirm"):
        release(vast, 19)
    assert session.instance_id == 19
    assert session.release_status == "failed"
    assert session._timings["release_utc"] == "unknown"
    vast.destroy_instance = lambda **kwargs: {"success": True}
    release(vast, 19)
    assert session.release_status == "released"
    assert session._timings["release_utc"] != "unknown"


def test_logging_and_stderr_failure_do_not_hide_the_created_contract(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path))
    session = RunLog("tools.start_qwen8b_r2", "qwen8b_r2_host", "transformers")
    blocked = tmp_path / "blocked"
    blocked.write_text("x")
    session.run_dir = blocked / "run"

    def fail_write(_):
        raise OSError("closed stderr")

    monkeypatch.setattr(run_log.sys, "stderr", SimpleNamespace(write=fail_write))
    vast = SimpleNamespace(
        create_instance=lambda **kwargs: {"success": True, "new_contract": 99}
    )
    assert (
        _wrap_create(start_sam.create_from_offers, session)(
            vast, [_offer()], disk_gb=80, label="test"
        )
        == 99
    )
    assert session.logging_errors


def test_all_log_sinks_remove_secret_assignments_and_untrusted_error_text(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path))
    session = RunLog("tools.start_qwen8b_r2", "qwen8b_r2_host", "transformers")
    secret = "synthetic-opaque-secret"
    sample = {
        "access_key_id": secret,
        "ssh_keys": [secret],
        "nested": {"description": "https://user:" + secret + "@example.test"},
        "note": "?api_key=" + secret + "&other=ok",
    }
    assert secret not in json.dumps(redact(sample))
    session.probe = {
        "raw": "R2_SECRET_ACCESS_KEY=" + secret,
        "gpus": [{"name": "token=" + secret}],
    }
    session._error("probe", RuntimeError(secret))
    session._write_entry()
    assert secret not in (session.run_dir / "entry.json").read_text()
    assert secret not in run_log.redact_text("?password=" + secret)


def test_probe_ignores_csv_banner_and_retains_partial_host_facts():
    stdout, _ = _live_probe_streams()
    gpu = stdout.splitlines()[0].replace("450.00 W", "[Not Supported]")
    banner = ", ".join(["banner"] * 13)
    parsed = parse_probe(banner + "\n" + gpu + "\nHOST\tCPU, model\t8\t\t\t\t\n")
    assert len(parsed["gpus"]) == 1
    assert parsed["gpus"][0]["power_limit"] == "unknown"
    assert parsed["host"]["cpu_model"] == "CPU, model"
    assert parsed["host"]["host_cpu_cores"] == "8"
    assert parsed["host"]["host_ram_total"] == "unknown"


def test_diagnostic_redaction_removes_private_key_body():
    text = "error: -----BEGIN OPENSSH PRIVATE KEY-----\nsynthetic-key-body\n"
    for ending in ("", "-----END OPENSSH PRIVATE KEY-----\nnext line"):
        cleaned = run_log.redact_text(text + ending)
        assert "synthetic-key-body" not in cleaned
        assert "PRIVATE KEY" not in cleaned


@pytest.mark.skipif(
    os.name != "posix", reason="executes the remote Linux shell locally"
)
def test_sampler_shell_writes_samples_and_stops(tmp_path, monkeypatch):
    import shutil
    import signal
    import time

    shell = shutil.which("sh")
    assert shell
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    binary = tmp_path / "nvidia-smi"
    binary.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '0, 1, 16, 4, 16, 20 W, 450 W, 210 MHz, 0 %, 10 MiB, 35' "
        "'1, 4, 16, 4, 16, 200 W, 450 W, 2000 MHz, 50 %, 100 MiB, 40'\n",
        encoding="utf-8",
    )
    binary.chmod(0o700)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    groups = []

    def local_ssh(remote, *, timeout, input=None):
        remote = remote.replace("/tmp/gv-gpu-", str(tmp_path / "gv-gpu-"))
        process = subprocess.Popen(
            [shell, "-c", remote],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        groups.append(process.pid)
        stdout, stderr = process.communicate(input, timeout=timeout)
        return subprocess.CompletedProcess(remote, process.returncode, stdout, stderr)

    session = RunLog("test", "test", "test", True)
    session.probe = {"status": "ok"}
    session._ssh_prefix = ["offline-shell"]
    monkeypatch.setattr(session, "_ssh", local_ssh)
    try:
        session.start_sampler()
        assert session._sampler_alive
        output = Path(
            session._remote_out.replace("/tmp/gv-gpu-", str(tmp_path / "gv-gpu-"))
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if output.exists() and len(output.read_text().splitlines()) >= 4:
                break
            time.sleep(0.05)
        assert output.exists(), "remote sampler must create its advertised output"
        assert len(output.read_text().splitlines()) >= 4
        session.stop_sampler()
        assert len(session._samples) >= 4
        assert {row["gpu_index"] for row in session._samples} == {0, 1}
        assert session._samples[0]["utilization_gpu"] == "0 %"
        assert (session.run_dir / "gpu-samples.jsonl").is_file()
        before = output.read_bytes()
        time.sleep(SAMPLE_INTERVAL + 0.1)
        assert output.read_bytes() == before
    finally:
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_missing_pcie_measurements_do_not_become_zero():
    missing = parse_sample_line(
        "100\tN/A, 16, N/A, 16, 20 W, 450 W, 210 MHz, 50 %, 10 MiB, 35", 0
    )
    assert (
        summarize_gpu_samples([missing], {}, 2)["run"]["low_pcie_gen_while_busy"]
        == "unknown"
    )
    known = dict(missing, t_unix=102, pcie_gen_current="1", pcie_gen_max="4")
    assert (
        summarize_gpu_samples([missing, known], {}, 2)["run"]["low_pcie_gen_while_busy"]
        == 1
    )


def test_retry_preserves_each_rental_and_resets_sampler_target(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    session = RunLog("test", "sam", "sam", True)
    offers = [_offer(), _offer(id=8, machine_id=4243)]
    session.note_create(offers, (), {"id": 7}, {"success": True, "new_contract": 10})
    first = session.run_dir
    session._probed = True
    session.probe = {"status": "ok", "gpus": [{"name": "first"}]}
    session._ssh_prefix = ["ssh", "first-host"]
    session.note_release()
    session.note_create(offers, (), {"id": 8}, {"success": True, "new_contract": 11})
    assert session.run_dir != first
    assert session.probe == "unknown"
    assert session._probed is False
    assert session._ssh_prefix is None
    assert session.release_status == "unknown"
    session.finish_once()
    first_entry = json.loads((first / "entry.json").read_text())
    last_entry = json.loads((session.run_dir / "entry.json").read_text())
    assert first_entry["instance_id"] == 10
    assert first_entry["release_status"] == "released"
    assert last_entry["instance_id"] == 11
    assert last_entry["hardware"]["offer"]["machine_id"] == 4243
    assert len(json.loads((session.root / "manifest.json").read_text())) == 2


def test_sampler_write_failure_keeps_score_and_final_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setattr(
        sys, "argv", ["launcher", "--eval-output", str(tmp_path / "eval")]
    )
    session = RunLog("test", "model", "test", True)

    def evaluate():
        report = tmp_path / "eval" / "model" / "validation"
        report.mkdir(parents=True)
        (report / "summary.json").write_text(json.dumps({"accuracy": 0.75}))
        return 0

    def fail():
        raise OSError("samples cannot be written")

    monkeypatch.setattr(session, "stop_sampler", fail)
    assert _wrap_eval(evaluate, session)() == 0
    session.finish_once()
    entry = json.loads((session.run_dir / "entry.json").read_text())
    assert entry["score"]["validation"]["accuracy"] == 0.75
    assert entry["eval_returncode"] == 0
    assert entry["logging_errors"]


def test_multi_gpu_samples_preserve_identity_and_busy_time():
    samples = [
        parse_sample_line(
            f"{stamp}\t{gpu}, {gen}, 16, 4, 16, 20 W, 450 W, 210 MHz, 50 %, 10 MiB, 35",
            0,
        )
        for stamp in (100, 102)
        for gpu, gen in ((0, 1), (1, 4))
    ]
    result = summarize_gpu_samples(samples, {}, 2)["run"]
    assert result["low_pcie_gen_while_busy"] == 0.5
    assert result["gpus"]["0"]["low_pcie_gen_while_busy"] == 1
    assert result["gpus"]["1"]["low_pcie_gen_while_busy"] == 0


def test_sample_jsonl_redacts_untrusted_values(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    session = RunLog("test", "test", "test", True)
    session._ingest(
        "100\t1, 16, 4, 16, 20 W, 450 W, token=private-value, 0 %, 10 MiB, 35"
    )
    text = (session.run_dir / "gpu-samples.jsonl").read_text()
    assert "private-value" not in text
    assert json.loads(text)["clock_sm"] == "redacted"


def test_busy_fraction_does_not_count_beyond_the_phase_end():
    low = parse_sample_line(
        "100\t1, 16, 4, 16, 20 W, 450 W, 210 MHz, 50 %, 10 MiB, 35", 0
    )
    high = dict(low, t_unix=102, pcie_gen_current="4")
    result = summarize_gpu_samples([low, high], {"test": (100, 102)}, 2)
    assert result["test"]["low_pcie_gen_while_busy"] == 1
    assert (
        summarize_gpu_samples([high], {"test": (102, 102)}, 2)["test"][
            "low_pcie_gen_while_busy"
        ]
        == "unknown"
    )


@pytest.mark.parametrize(
    "owns,offset,phase",
    [
        (False, 0, "run"),
        (True, "unknown", "run"),
        (True, 0, "validation"),
    ],
)
def test_sample_phase_requires_owned_report_and_known_clock(
    tmp_path, monkeypatch, owns, offset, phase
):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    session = RunLog("test", "model", "test", True)
    session.eval_output = str(tmp_path / "eval")
    session._owns_eval_output = owns
    session._clock_offset = offset
    report = tmp_path / "eval" / "model" / "validation"
    report.mkdir(parents=True)
    (report / "summary.json").write_text(
        json.dumps(
            {
                "started_at_unix": 99,
                "finished_at_unix": 102,
            }
        )
    )
    session._ingest("100\t1, 16, 4, 16, 20 W, 450 W, 210 MHz, 50 %, 10 MiB, 35")
    assert set(session._sample_summary) == {phase}


_PROBE_STDOUT = (
    "RTX 4090, 0000:81:00.0, 1, 4, 8, 16, 450.00 W, 210 MHz, 3105 MHz, "
    "10251 MHz, 550.90, 24564 MiB, 0 %\n"
    "HOST\tAMD EPYC\t16\t32000000 kB\t12.8\tunknown\tunknown\n"
)


@pytest.mark.parametrize("model", ["clip", "siglip2", "videojev", "decision3"])
def test_native_execute_logs_hardware_duration_score_and_person(
    tmp_path, monkeypatch, model
):
    run_log._SESSION = None
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("GOODVISION_OPERATOR", "colleague")
    out = tmp_path / "eval" / model
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_run.py",
            "--model",
            model,
            "--execute",
            "--output",
            str(out),
            "--phase",
            "test",
        ],
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=_PROBE_STDOUT, stderr="")

    monkeypatch.setattr(run_log.subprocess, "run", fake_run)
    arm_if_launcher(str(REPO / "tools/unified_benchmark/native_run.py"))
    session = run_log._SESSION
    assert session is not None and session.model == model
    assert session.local_probe is True
    assert not out.exists()
    (out / "test").mkdir(parents=True)
    (out / "test" / "report.json").write_text(
        json.dumps(
            {
                "accuracy": 0.5,
                "final_unknown_questions": 1,
                "final_missing_unique_units": 0,
                "test_attempt_latency": {
                    "p50_seconds": 1.2,
                    "p95_seconds": 2.2,
                    "n": 4,
                    "source": "native",
                },
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    session.note_eval_end(0)
    session.finish_once()
    entry = json.loads((session.run_dir / "entry.json").read_text(encoding="utf-8"))
    assert entry["who"]["display"] == "colleague"
    assert entry["eval_returncode"] == 0
    assert entry["model"] == model
    assert entry["engine"] == NATIVE_MODELS[model][2]
    assert isinstance(entry["timings"]["wall_seconds"], float)
    assert entry["hardware"]["probe"]["status"] == "ok"
    assert entry["hardware"]["probe"]["gpus"][0]["pcie_gen_current"] == "1"
    assert entry["hardware"]["probe"]["host"]["cpu_model"] == "AMD EPYC"
    assert entry["score"]["test"]["accuracy"] == 0.5
    assert entry["score"]["test"]["p50_seconds"] == 1.2
    rows = json.loads((session.root / "manifest.json").read_text(encoding="utf-8"))
    assert rows[-1]["who"]["operator_env"] == "colleague"
    assert rows[-1]["score"]["test"]["final_missing_unique_units"] == 0
    run_log._SESSION = None


def test_native_log_stays_off_without_execute_or_for_other_models(
    tmp_path, monkeypatch
):
    run_log._SESSION = None
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["native_run.py", "--model", "clip", "--output", str(tmp_path / "out")],
    )
    arm_if_launcher("tools/unified_benchmark/native_run.py")
    assert run_log._SESSION is None
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_run.py",
            "--model",
            "luna",
            "--execute",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    arm_if_launcher("tools/unified_benchmark/native_run.py")
    assert run_log._SESSION is None
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_run.py",
            "--model",
            "qwen8b",
            "--execute",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    arm_if_launcher("tools/unified_benchmark/native_run.py")
    assert run_log._SESSION is None


@pytest.fixture
def native_log_env(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log, "_SESSION", None)
    monkeypatch.setattr(run_log.atexit, "register", lambda callback: None)
    monkeypatch.setattr(run_log, "_git_user", lambda: "tester")
    monkeypatch.setattr(run_log, "_git_commit", lambda: "test-revision")
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "logs"))
    output = tmp_path / "eval"
    monkeypatch.setattr(
        sys,
        "argv",
        ["native_run.py", "--model", "clip", "--execute", "--output", str(output)],
    )
    monkeypatch.setattr(
        run_log.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, _PROBE_STDOUT, ""),
    )
    return output


@pytest.mark.parametrize(
    ("problem", "code"),
    [
        (None, 0),
        (SystemExit(0), 0),
        (SystemExit(None), 0),
        (SystemExit(7), 7),
        (SystemExit("invalid"), 1),
        (FileExistsError("another invocation created output"), 1),
        (KeyboardInterrupt(), 130),
    ],
)
def test_native_hook_preserves_exit_and_rejects_failed_scores(
    native_log_env, problem, code
):
    output = native_log_env

    def write_report():
        (output / "test").mkdir(parents=True)
        (output / "test/report.json").write_text('{"accuracy": 0.5}')
        (output / "run.json").write_text('{"model": "clip", "slots": 12}')

    script = compile(
        "write_report()\nif problem is not None: raise problem",
        str(REPO / "tools/unified_benchmark/native_run.py"),
        "exec",
    )
    globals_dict = {"write_report": write_report, "problem": problem}
    if problem is None:
        runpy._run_code(script, globals_dict, mod_name="__main__")
    else:
        with pytest.raises(type(problem)) as caught:
            runpy._run_code(script, globals_dict, mod_name="__main__")
        assert caught.value is problem
    session = run_log._SESSION
    entry_path = session.run_dir / "entry.json"
    entry = json.loads(entry_path.read_text())
    assert entry["eval_returncode"] == code
    assert entry["timings"]["eval_end_utc"] != "unknown"
    if code == 0:
        assert entry["score"]["test"]["accuracy"] == 0.5
        assert entry["score"]["concurrency_slots"] == 12
    else:
        assert entry["score"] == "unknown"
    # The native result inventory must not acquire logs after it is sealed.
    assert sorted(p.relative_to(output).as_posix() for p in output.rglob("*")) == [
        "run.json",
        "test",
        "test/report.json",
    ]
    before = entry_path.read_bytes()
    session.finish_once()
    assert entry_path.read_bytes() == before


def test_native_reused_output_keeps_old_score_unknown(native_log_env):
    output = native_log_env
    (output / "test").mkdir(parents=True)
    (output / "test/report.json").write_text('{"accuracy": 1}')
    session = arm_if_launcher("tools/unified_benchmark/native_run.py")
    run_log.finish_native(session, 0)
    assert session.score == "unknown"
    assert not (output / "clip").exists()


def test_score_layouts_never_borrow_another_model_report(tmp_path):
    (tmp_path / "test").mkdir()
    (tmp_path / "test/report.json").write_text('{"accuracy": 1}')
    assert summarize_score(tmp_path, "qwen8b_r2_host") == "unknown"
    (tmp_path / "clip/test").mkdir(parents=True)
    (tmp_path / "clip/test/summary.json").write_text('{"accuracy": 0}')
    assert summarize_score(tmp_path, "clip", native=True)["test"]["accuracy"] == 1
    (tmp_path / "test/report.json").write_text("invalid json")
    assert summarize_score(tmp_path, "clip", native=True) == "unknown"


def test_native_repeated_flags_match_argparse(native_log_env, monkeypatch):
    output = native_log_env
    sys.argv += [
        "--model=siglip2",
        "--output",
        str(output / "actual"),
        "--phase=test",
        "--phase",
        "both",
    ]
    session = arm_if_launcher("tools/unified_benchmark/native_run.py")
    assert session.model == "siglip2"
    assert session.eval_output == str(output / "actual")
    assert session.eval_phase == "both"
    monkeypatch.setattr(run_log, "_SESSION", None)
    sys.argv += ["--model=luna"]
    assert arm_if_launcher("tools/unified_benchmark/native_run.py") is None


def test_native_help_has_no_logging_side_effects(native_log_env):
    sys.argv += ["--help"]
    assert arm_if_launcher("tools/unified_benchmark/native_run.py") is None
    assert not native_log_env.parent.joinpath("logs").exists()


@pytest.mark.parametrize(
    "failure", [OSError("missing sh"), subprocess.TimeoutExpired("sh", 20)]
)
def test_native_probe_failure_preserves_cli_outcome(
    native_log_env, monkeypatch, failure
):
    session = arm_if_launcher("tools/unified_benchmark/native_run.py")

    def fail_probe(command, **kwargs):
        assert command == ["sh", "-c", _PROBE]
        assert kwargs["timeout"] == 20
        raise failure

    monkeypatch.setattr(run_log.subprocess, "run", fail_probe)
    run_log.finish_native(session, 3)
    entry = json.loads((session.run_dir / "entry.json").read_text())
    assert entry["hardware"]["probe"] == "unknown"
    assert entry["eval_returncode"] == 3
    assert entry["logging_errors"]


def test_native_module_failure_is_logged_without_inference(native_log_env, monkeypatch):
    # This reaches the real CLI and fails before execute or any model loading.
    output = native_log_env
    sys.argv += [
        "--bundle",
        str(output.parent / "missing-bundle"),
        "--deployment",
        str(output.parent / "missing-deployment"),
        "--deployment-sha256",
        "unused",
        "--authorization-note",
        "offline test",
    ]
    monkeypatch.delitem(
        sys.modules, "tools.unified_benchmark.native_run", raising=False
    )
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_XET"):
        monkeypatch.setenv(name, "1")
    with pytest.raises(FileNotFoundError):
        runpy.run_module("tools.unified_benchmark.native_run", run_name="__main__")
    session = run_log._SESSION
    entry = json.loads((session.run_dir / "entry.json").read_text())
    assert entry["eval_returncode"] == 1
    assert entry["score"] == "unknown"
    assert not output.exists()


def test_native_logging_write_failure_preserves_exception(native_log_env, monkeypatch):
    def fail_write(self):
        raise OSError("disk full")

    monkeypatch.setattr(RunLog, "_write_entry", fail_write)
    problem = RuntimeError("original failure")
    script = compile(
        "raise problem", str(REPO / "tools/unified_benchmark/native_run.py"), "exec"
    )
    with pytest.raises(RuntimeError) as caught:
        runpy._run_code(script, {"problem": problem}, mod_name="__main__")
    assert caught.value is problem
    assert run_log._SESSION.eval_returncode == 1


def test_other_native_run_filename_is_not_armed(native_log_env):
    assert arm_if_launcher(str(native_log_env.parent / "native_run.py")) is None
    assert run_log._SESSION is None


def test_native_nonzero_probe_is_unknown(native_log_env, monkeypatch):
    session = arm_if_launcher("tools/unified_benchmark/native_run.py")
    monkeypatch.setattr(
        run_log.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, _PROBE_STDOUT, ""),
    )
    run_log.finish_native(session, 0)
    assert session.probe["status"] == "unknown"
    assert session.probe["gpus"] == "unknown"
    assert session.logging_errors


def test_native_log_tracks_abbreviations_and_parser_options(
    native_log_env, monkeypatch
):
    from tools.unified_benchmark import native_run

    class Parsed(Exception):
        pass

    def check_options(parser, argv):
        actual = set(parser._option_string_actions)
        logged = set(run_log._native_log_parser()._option_string_actions)
        assert actual == logged
        raise Parsed

    monkeypatch.setattr(native_run.argparse.ArgumentParser, "parse_args", check_options)
    with pytest.raises(Parsed):
        native_run.main([])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_run.py",
            "--mod=decision3",
            "--exec",
            "--out",
            str(native_log_env),
            "--ph=test",
        ],
    )
    session = arm_if_launcher("tools/unified_benchmark/native_run.py")
    assert session.model == "decision3"
    assert session.eval_output == str(native_log_env)
    assert session.eval_phase == "test"


def test_native_log_leaves_invalid_cli_diagnostics_to_runner(native_log_env, capsys):
    sys.argv += ["--p", "test"]  # ambiguous between --port and --phase
    assert arm_if_launcher("tools/unified_benchmark/native_run.py") is None
    assert capsys.readouterr().err == ""


def test_native_resume_waiting_for_new_output_never_claims_score(native_log_env):
    output = native_log_env
    sys.argv += ["--resume"]
    session = arm_if_launcher("tools/unified_benchmark/native_run.py")
    assert not output.exists()
    # A concurrent invocation can create this while resume waits for its lock.
    (output / "test").mkdir(parents=True)
    (output / "test/report.json").write_text('{"accuracy": 1}')
    run_log.finish_native(session, 0)
    assert session.score == "unknown"
