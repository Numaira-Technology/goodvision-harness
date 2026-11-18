"""A failed console sink must not prevent owned-instance destruction."""

import contextlib
import io
import os
from types import SimpleNamespace

import pytest
from test_qwen8b_release import fake_runtime

from tools import start_qwen8b_r2 as host
from tools import start_qwen8b_vllm as vllm
from tools import start_sam
from tools.qwen8b import eval_local


@contextlib.contextmanager
def pipe_stdout():
    reader, writer = os.pipe()
    stream = os.fdopen(writer, "w", encoding="utf-8", buffering=1)
    try:
        yield reader, stream
    finally:
        try:
            os.close(reader)
        except OSError:
            pass
        try:
            stream.close()
        except OSError:
            pass


@pytest.mark.parametrize("failure_point", ["before", "after", "closed-stream"])
def test_release_ignores_console_failure_and_restores_timeout(failure_point):
    calls = []
    client = SimpleNamespace(timeout=(3, 4))
    with pipe_stdout() as (reader, stream):
        if failure_point == "before":
            os.close(reader)
        elif failure_point == "closed-stream":
            stream = io.StringIO()
            stream.close()

        def destroy(*, id):
            calls.append((id, client.timeout))
            if failure_point == "after":
                os.close(reader)
            return {"success": True}

        with contextlib.redirect_stdout(stream):
            start_sam.release_instance(
                SimpleNamespace(client=client, destroy_instance=destroy), 123
            )
    assert calls == [(123, (5, 20))]
    assert client.timeout == (3, 4)


@pytest.mark.parametrize(
    "failure,error,message",
    [
        ("rejected", RuntimeError, "did not confirm instance destruction"),
        ("malformed", RuntimeError, "did not confirm instance destruction"),
        ("exception", OSError, "synthetic provider failure"),
    ],
)
def test_console_failure_does_not_hide_provider_failure(failure, error, message):
    calls = []
    client = SimpleNamespace(timeout=(3, 4))

    def destroy(*, id):
        calls.append((id, client.timeout))
        if failure == "exception":
            raise OSError("synthetic provider failure")
        return {"success": False} if failure == "rejected" else None

    with pipe_stdout() as (reader, stream):
        os.close(reader)
        with contextlib.redirect_stdout(stream), pytest.raises(error, match=message):
            start_sam.release_instance(
                SimpleNamespace(client=client, destroy_instance=destroy), 123
            )
    assert calls == [(123, (5, 20))]
    assert client.timeout == (3, 4)


@pytest.mark.parametrize("module", [host, vllm], ids=["hf", "vllm"])
def test_stdout_consumer_disappears_after_evaluation(tmp_path, monkeypatch, module):
    vast, status, stopped = fake_runtime(monkeypatch, tmp_path, module, None)
    with pipe_stdout() as (reader, stream):
        def evaluate(argv):
            # A shell pipeline/log consumer exits after the instance was created.
            os.close(reader)
            return 0

        monkeypatch.setattr(eval_local, "main", evaluate)
        with contextlib.redirect_stdout(stream):
            result = module.main()
    assert result == 0
    assert stopped == [True]
    assert vast.destroyed == [123] and not vast.alive
    assert vast.client.timeout == (3, 4)
    assert "release-unconfirmed" not in status.read_text()
