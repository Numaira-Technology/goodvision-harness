"""Opt-in ownership for API tests that suppress StreamingSession.start."""

from contextlib import ExitStack

from goodvision_harness.pipeline import StreamingSession

WORKER_CLOSE_SECONDS = 1.0


def _close_worker(worker):
    worker.close(timeout_seconds=WORKER_CLOSE_SECONDS)
    assert worker._closing
    assert not worker._thread.is_alive(), "fake-start identity worker did not stop"


class FakeSessionOwner:
    def __init__(self, runtime, monkeypatch, request):
        self.runtime = runtime
        self.workers = []
        self.sessions = []
        # Pytest keeps body failures separate from teardown failures, and still
        # runs other owners' finalizers if this owner's cleanup fails.
        request.addfinalizer(self.close)

        def start(session):
            assert runtime._session is session
            worker = runtime._identity_worker
            assert worker is not None
            # Retain the actual owner before stop/replacement can clear it.
            self.workers.append(worker)
            self.sessions.append(session)
            assert session._thread is None
            # No real session thread, capture, inference or on_end callback.

        monkeypatch.setattr(StreamingSession, "start", start)

    def close(self):
        try:
            self.runtime.stop()
        finally:
            # stop() has released the runtime locks needed by review callbacks.
            # ExitStack attempts every owned close even if an earlier one fails.
            with ExitStack() as cleanup:
                for worker in self.workers:
                    cleanup.callback(_close_worker, worker)
