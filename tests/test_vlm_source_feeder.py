import time
from threading import Event, Thread

import pytest

from tools.vlm_benchmark.source_feeder import TimedSourceFeeder


class Branch:
    def __init__(self, accepted=True):
        self.calls = []
        self.accepted = accepted

    def offer(self, frame, timestamp, tick):
        self.calls.append((frame, timestamp, tick))
        return self.accepted


def test_slow_monitor_does_not_delay_source_or_drop_notifications():
    entered, unblock = Event(), Event()
    notifications = []

    class Observer:
        def sample(self, *sample):
            entered.set()
            assert unblock.wait(2)
            notifications.append(sample)

    branch = Branch()
    feeder = TimedSourceFeeder(
        [branch],
        [[b"a", b"b"]],
        [(i * 0.04, [(0, i)]) for i in range(5)],
        time.monotonic(),
        2.5,
    )
    feeder.start()
    consumer = Thread(target=lambda: feeder.notify(Observer()))
    try:
        # Ensure the initial notification exists before starting the consumer.
        deadline = time.monotonic() + 1
        while not branch.calls and time.monotonic() < deadline:
            time.sleep(0.001)
        consumer.start()
        assert entered.wait(1)
        feeder.thread.join(1)
        feeder.check()
        assert not feeder.thread.is_alive()
        assert len(branch.calls) == 5  # Consumer is still blocked.
        assert not notifications
    finally:
        unblock.set()
        if consumer.ident:
            consumer.join(2)
        feeder.close()
    assert notifications == [(0, i, True) for i in range(5)]
    assert branch.calls == [
        (b"a" if i % 2 == 0 else b"b", i / 2.5, i) for i in range(5)
    ]


def test_source_error_propagates_to_benchmark():
    class Broken:
        def offer(self, *_):
            raise ValueError("source failed")

    feeder = TimedSourceFeeder(
        [Broken()], [[b"a"]], [(0, [(0, 0)])], time.monotonic(), 2.5
    )
    feeder.start()
    feeder.thread.join(1)
    with pytest.raises(ValueError, match="source failed"):
        feeder.check()
    feeder.close()


def test_expired_inputs_still_count_as_missed_and_are_not_caught_up():
    branch = Branch(False)
    feeder = TimedSourceFeeder(
        [branch], [[b"a"]], [(0, [(0, 0)])], time.monotonic() - 1, 2.5
    )
    feeder.start()
    feeder.thread.join(1)
    feeder.check()
    feeder.close()
    assert feeder.missed == 1 and feeder.missed_by_stream == [1]
    assert not branch.calls
    assert feeder.release_lags[0] >= 1


def test_cancel_interrupts_wait_before_future_source_offer():
    branch = Branch()
    feeder = TimedSourceFeeder(
        [branch], [[b"a"]], [(30, [(0, 0)])], time.monotonic(), 2.5
    )
    feeder.start()
    feeder.close()
    assert not feeder.thread.is_alive()
    assert not branch.calls


def test_slow_camera_cannot_hide_late_siblings_in_one_release(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(
        "tools.vlm_benchmark.source_feeder.time.monotonic", lambda: clock[0]
    )

    class Slow(Branch):
        def offer(self, *args):
            clock[0] += 0.5
            return super().offer(*args)

    first, second = Slow(), Branch()
    feeder = TimedSourceFeeder(
        [first, second], [[b"a"]], [(0, [(0, 0), (1, 0)])], 100.0, 2.5
    )
    feeder._run()
    feeder.check()
    assert len(first.calls) == 1
    assert not second.calls
    assert feeder.missed == 1 and feeder.missed_by_stream == [0, 1]
    assert max(feeder.release_lags) == 0.5
