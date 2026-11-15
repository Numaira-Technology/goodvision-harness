"""Offline tests for the Qwen Vast run log. No GPU and no Vast API."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import start_sam
from tools.vast_run_log import (
    RunLog, _wrap_create, _wrap_eval, logs_directory, offer_fields, parse_probe,
    redact, summarize_score,
)

REPO = Path(__file__).resolve().parents[1]


def _offer(**extra):
    body = {
        "id": 7, "machine_id": 4242, "host_id": 9, "geolocation": "California, US",
        "dph_total": 0.42, "gpu_name": "RTX_4090", "num_gpus": 1, "gpu_ram": 0,
        "gpu_frac": 0.5, "ssh_host": "ssh.example", "ssh_port": 2222,
        "api_key": "vast-secret-value", "ssh_key": "-----BEGIN PRIVATE KEY-----\nabc",
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
    text = "RTX 4090, 0000:81:00.0, 4, 4, 8, 16, 450.00 W, 2520 MHz, 3105 MHz, 10251 MHz, 550.90, 24564 MiB\n"
    parsed = parse_probe(text)
    assert parsed["status"] == "ok"
    assert parsed["gpus"][0]["pcie_gen_current"] == "4"
    assert parsed["gpus"][0]["power_limit"] == "450.00 W"
    assert parse_probe("nope")["status"] == "unknown"
    assert parse_probe("")["gpus"] == "unknown"


def test_score_summary_preserves_zero_and_marks_missing_unknown(tmp_path):
    alias = "qwen8b_r2_host"
    split = tmp_path / alias / "validation"
    split.mkdir(parents=True)
    (split / "summary.json").write_text(json.dumps({
        "accuracy": 0, "correct": 0, "planned_questions": 24,
        "final_unknown_questions": 2, "final_missing_unique_units": 0,
        "test_attempt_latency": {"n": 3, "source": "transport"},
    }), encoding="utf-8", newline="\n")
    (tmp_path / "run-manifest.json").write_text(
        json.dumps({"slots_per_model": 12}), encoding="utf-8", newline="\n")
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
    chosen = _offer(id=2, machine_id=20, dph_total=0.4, geolocation="B", gpu_name="RTX_4090")
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
    saved = json.loads((session.run_dir / "vast-offer.json").read_text(encoding="utf-8"))
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
        "machine_id": 20, "gpu_name": "RTX_4090", "dph_total": 0.4, "geolocation": "B",
    }
    assert rows[0]["who"]["operator_env"] == "colleague"
    assert rows[0]["score"] == "unknown"
    report = tmp_path / "report"
    (report / "qwen8b_r2_host" / "validation").mkdir(parents=True)
    (report / "qwen8b_r2_host" / "validation" / "summary.json").write_text(json.dumps({
        "accuracy": 0, "final_unknown_questions": 1, "final_missing_unique_units": 0,
    }), encoding="utf-8", newline="\n")
    session.eval_output = str(report)
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
    first.note_create([_offer(id=5, machine_id=50)], (), {"id": 5}, {"success": True, "new_contract": 5})
    second.note_create([_offer(id=6, machine_id=60, geolocation="EU")], (), {"id": 6}, {"success": True, "new_contract": 6})
    rows = json.loads((first.root / "manifest.json").read_text(encoding="utf-8"))
    assert [row["gpu"]["machine_id"] for row in rows] == [50, 60]
    assert json.loads((first.run_dir / "vast-offer.json").read_text(encoding="utf-8"))["machine_id"] == 50
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


def test_score_lands_in_the_model_report_without_creating_a_missing_output(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODVISION_RUN_LOG_ROOT", str(tmp_path / "series"))
    report_root = tmp_path / "eval-out"
    alias = "qwen8b_r2_host"
    (report_root / alias / "test").mkdir(parents=True)
    (report_root / alias / "test" / "summary.json").write_text(json.dumps({
        "accuracy": 0.25, "final_unknown_questions": 4, "final_missing_unique_units": 1,
        "test_attempt_latency": {"p50_seconds": 1.5, "p95_seconds": 3.5, "n": 10, "source": "transport"},
    }), encoding="utf-8", newline="\n")
    session = RunLog("tools.start_qwen8b_r2", alias, "transformers")
    session.eval_output = str(report_root)
    session.mark_rent_start()
    session.note_create([_offer()], (), {"id": 7}, {"success": True, "new_contract": 11})
    session.finish_once()
    copied = report_root / alias / "logs" / session.run_id / "entry.json"
    entry = json.loads(copied.read_text(encoding="utf-8"))
    assert entry["score"]["test"]["accuracy"] == 0.25
    assert entry["score"]["test"]["p50_seconds"] == 1.5
    assert entry["timings"]["rent_start_utc"] != "unknown"
    assert isinstance(entry["timings"]["wall_seconds"], float)
    mirror = json.loads((report_root / alias / "logs" / "manifest.json").read_text(encoding="utf-8"))
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
        [sys.executable, "-m", module, "--eval-bundle", str(bundle), "--eval-output", str(output)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
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
