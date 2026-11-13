"""Contract fixture wait measurements cannot be treated as production drops."""

from types import SimpleNamespace

import benchmark_runtime_fixtures as fixtures
import pytest


@pytest.mark.parametrize("fail_offer", [False, True])
def test_fixture_diagnostics_preserve_group_waits_without_inventing_samples(
    monkeypatch, fail_offer,
):
    clock = [100.0]
    monkeypatch.setattr(fixtures, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    delivered = []

    class Branch:
        config = SimpleNamespace(stream_session_id="synthetic")

        def offer(self, frame, timestamp, tick):
            if fail_offer:
                raise RuntimeError("synthetic offer failure")
            delivered.append((frame, timestamp, tick))
            return True

    class Stop:
        def wait(self, delay):
            clock[0] += delay + 0.6
            return False

        def is_set(self):
            return False

    feeder = fixtures.ContractSourceFeeder(
        [Branch(), Branch()], [[b"a", b"b"]],
        [(0, [(0, 0), (1, 0)]), (0.4, [(0, 1), (1, 1)])], 100.0, 2.5,
    )
    feeder.stop = Stop()
    feeder._run()
    report = feeder.diagnostics()
    assert report["scope"] == "contract fixture release-group waits only"
    assert report["release_groups_planned"] == 2
    assert report["release_groups_measured"] == (1 if fail_offer else 2)
    assert report["release_group_excess_wait_seconds"] == pytest.approx(
        [0.6] if fail_offer else [0.6, 0.6]
    )
    assert report["planned_sample_count"] == 4
    assert report["sample_measurements_available"] is False
    assert report["missed"] == 0 and report["missed_by_stream"] == [0, 0]
    assert report["offer_rejected"] == [0, 0]
    assert report["feeder_error_type"] == ("RuntimeError" if fail_offer else None)
    assert "samples" not in report and "derived_drop_count" not in report
    assert len(delivered) == (0 if fail_offer else 4)
