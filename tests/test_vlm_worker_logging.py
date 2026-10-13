"""Fresh spawn must retain safe model validation and superseded-result logs."""

import contextlib
import json
import multiprocessing as mp

from goodvision_harness.latency_trace import configure_worker_timing, log_timing
from goodvision_harness.scene_model import _reply_diagnostics


def emit_worker_evidence(path):
    with open(path, "w", encoding="utf-8") as stream:
        with contextlib.redirect_stderr(stream):
            configure_worker_timing()
            configure_worker_timing()  # Reinitialization must not duplicate records.
            log_timing(
                "vlm_response_validation",
                **_reply_diagnostics('```json\n{"PRIVATE_SECRET":false,}\n```'),
            )
            log_timing("vlm_information_report_discarded", error_code="timeout")


def test_fresh_spawn_retains_safe_evidence_once(tmp_path):
    path = tmp_path / "worker.log"
    process = mp.get_context("spawn").Process(
        target=emit_worker_evidence, args=(str(path),),
    )
    process.start()
    try:
        process.join(15)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
    text = path.read_text(encoding="utf-8")
    assert "PRIVATE_SECRET" not in text
    lines = text.splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0][lines[0].index("{"):])
    assert first["json_wrapper"] == "complete_fence"
    assert first["diagnostic_json_reason"] == "invalid_json_syntax"
    assert first["json_error_line"] == 1
    assert '"error_code":"timeout"' in lines[1]
