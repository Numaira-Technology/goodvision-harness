import pytest

from goodvision_harness.stream_timing import SourceCadence


def test_source_pts_keep_all_frames_despite_bursty_arrival():
    cadence = SourceCadence(15)
    timestamps = []
    for i in range(150):
        now = 10 + (i // 15) + (i % 15) * 0.003
        timestamp, accepted = cadence.select(now, i / 15)
        assert accepted
        timestamps.append(timestamp)
    assert abs(timestamps[-1] - 149 / 15) < 1e-8


def test_high_rate_source_is_sampled_on_media_clock():
    cadence = SourceCadence(15)
    chosen = [i for i in range(300) if cadence.select(i / 30, i / 30)[1]]
    assert chosen == list(range(0, 300, 2))


def test_pts_restart_and_reconnect_keep_strictly_increasing_time():
    cadence = SourceCadence(15)
    first = [cadence.select(10 + i / 15, 900 + i / 15)[0] for i in range(5)]
    assert not cadence.select(11, 900 + 4 / 15)[1]
    restarted, accepted = cadence.select(11, 0)
    assert accepted and restarted > first[-1]
    assert cadence.discontinuities == 1
    cadence.reconnect()
    after, accepted = cadence.select(12, 0)
    assert accepted and after > restarted


def test_arrival_fallback_does_not_accumulate_interval_drift():
    cadence = SourceCadence(15)
    accepted = sum(
        cadence.select(i / 15 + (0.0004 if i % 2 else 0), None)[1] for i in range(150)
    )
    assert accepted == 150


def test_rtp_forward_wrap_jumps_do_not_expire_recent_evidence():
    cadence = SourceCadence(15)
    stamps = []
    # Reproduce repeated 32-bit / 90 kHz jumps seen on the live WHEP feed.
    wrap = 2**32 / 90000
    for index in range(150):
        now = 10 + index / 15
        pts = index / 15 + (index // 30) * wrap
        timestamp, selected = cadence.select(now, pts)
        assert selected
        stamps.append(timestamp)
    assert cadence.discontinuities == 4
    assert stamps == pytest.approx([index / 15 for index in range(150)])
    assert stamps[31] - stamps[29] < 1  # In-flight tracking evidence stays fresh.


def test_real_receive_outage_preserves_elapsed_media_time():
    cadence = SourceCadence(15)
    cadence.select(10, 900)
    timestamp, selected = cadence.select(30, 920)
    assert selected and timestamp == 20
    assert cadence.discontinuities == 0


def test_corrected_timestamp_returning_to_original_clock_keeps_progress():
    cadence = SourceCadence(15)
    cadence.select(10, 0)
    jump, _ = cadence.select(10.1, 47722)
    restored, selected = cadence.select(10.2, .2)
    assert selected and jump < restored < 1
    assert cadence.discontinuities == 2
