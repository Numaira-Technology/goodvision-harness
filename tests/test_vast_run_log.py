"""Offline tests for the Qwen Vast run log. No GPU and no Vast API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import start_sam
from tools import vast_run_log as run_log
from tools.vast_run_log import (
    _HOST_FIELDS,
    _PROBE,
    RunLog,
    _wrap_create,
    _wrap_eval,
    logs_directory,
    offer_fields,
    parse_probe,
    redact,
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
    assert session.probe["raw"].startswith("Welcome to vast.ai")
    assert "Have fun!" in session.probe["raw"]
    assert "3200000" not in session.probe["raw"]
    saved = json.loads((session.run_dir / "entry.json").read_text(encoding="utf-8"))
    assert saved["hardware"]["probe"]["host"]["container_cpu_limit"].startswith(
        "32 cores"
    )
    assert "Welcome to vast.ai" in saved["hardware"]["probe"]["raw"]


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
