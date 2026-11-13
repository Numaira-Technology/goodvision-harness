"""Embedded callers cannot accidentally disable queue backpressure."""

import queue

import pytest

from goodvision_harness.neural_service import _BatchWorker, create_app


@pytest.mark.parametrize("capacity", [0, -1, False, True, 1.5, "1", None])
@pytest.mark.parametrize("entrypoint", ["worker", "app"])
def test_queue_capacity_must_be_a_positive_integer(capacity, entrypoint):
    with pytest.raises(ValueError, match="max_queue must be a positive integer"):
        if entrypoint == "worker":
            _BatchWorker(lambda pixels, labels: [], 1, max_queue=capacity)
        else:
            create_app(object(), max_queue=capacity)


@pytest.mark.parametrize("capacity", [1, 2, 32])
def test_valid_capacity_rejects_one_more_pending_request(capacity):
    worker = _BatchWorker(lambda pixels, labels: [], 1, max_queue=capacity)
    assert not worker.thread.is_alive()
    pending = [worker.submit([index], ["fall"], lambda *args: None)
               for index in range(capacity)]
    assert len(pending) == worker.pending.qsize() == capacity
    with pytest.raises(queue.Full):
        worker.submit([capacity], ["fall"], lambda *args: None)
    assert worker.pending.qsize() == capacity
    assert all(not future.done() for future in pending)
