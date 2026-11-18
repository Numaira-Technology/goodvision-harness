"""Ancillary latency logging must preserve the caller's result, without I/O."""
import contextlib
import io

import pytest

from tools import start_qwen8b_vllm as launcher, vast_run_log
from tools.latency_investigation import host_facts


@pytest.mark.parametrize("helper,args,recorder", [
    ("_record_setup", ("synthetic", 1.0), "record_setup"),
    ("_record_client_access", ("synthetic.invalid", 22), "record_client_access"),
    ("_metrics_finish", ("unknown",), "record_metrics_diff"),
])
@pytest.mark.parametrize("sink_kind", ["closed", "broken_pipe"])
@pytest.mark.parametrize("caller_fails", [False, True])
def test_latency_warning_preserves_caller_outcome(
    monkeypatch, helper, args, recorder, sink_kind, caller_fails,
):
    calls = []

    def recording_failed(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("synthetic diagnostic failure")

    class BrokenPipe:
        def write(self, text):
            raise BrokenPipeError("synthetic closed reader")

    sink = io.StringIO() if sink_kind == "closed" else BrokenPipe()
    if sink_kind == "closed":
        sink.close()
    monkeypatch.setattr(vast_run_log, recorder, recording_failed)
    monkeypatch.setattr(host_facts, "client_access", lambda **kwargs: {})
    monkeypatch.setattr(launcher, "_metrics_snapshot", lambda: "unknown")
    original = RuntimeError("original evaluation failure")

    def evaluate():
        try:
            if caller_fails:
                raise original
            return 17
        finally:
            getattr(launcher, helper)(*args)

    with contextlib.redirect_stderr(sink):
        if caller_fails:
            with pytest.raises(RuntimeError) as caught:
                evaluate()
            assert caught.value is original
        else:
            assert evaluate() == 17
    assert len(calls) == 1
