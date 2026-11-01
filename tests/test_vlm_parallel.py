import json
import time
from dataclasses import replace
from queue import Full, Queue
from threading import Event, Thread

import pytest
from test_sam_display_fusion import batch, frame, setup, vlm
from test_vlm_independent import SourceSample, config, job, jpeg, utc_now, verdict

from goodvision_harness.scene_model import _ResponseValidationError
from goodvision_harness.vlm_compact_model import (
    CompactGroundingInspector,
    EventJSONInspector,
)
from goodvision_harness.vlm_independent import VLMProcessConfig, run_vlm_process


def envelope(raw):
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]
    }


def test_event_json_strict_coverage_and_unknown():
    j = job(config(), 1.2, 18)
    votes = EventJSONInspector._parse(j, envelope({"smoking": None, "fall": False}))
    assert votes[0].detected is None and votes[1].detected is False
    assert all(not v.subject_ids and not v.subject_boxes for v in votes)
    for raw in (
        {"fall": False},
        {"smoking": 0, "fall": False},
        {"smoking": False, "fall": False, "extra": 0},
    ):
        with pytest.raises(_ResponseValidationError):
            EventJSONInspector._parse(j, envelope(raw))


@pytest.mark.parametrize(
    "raw",
    [
        {"no_events": True},
        {"no_events": -1},
        {"no_events": 1.1},
        {"no_events": 0.9, "verdicts": []},
    ],
)
def test_compact_negative_rejects_invalid_or_mixed_output(raw):
    with pytest.raises(_ResponseValidationError):
        CompactGroundingInspector._parse(job(config(), 1.2, 18), envelope(raw))


def test_compact_negative_cannot_clear_a_person_or_replace_required_followup():
    j = job(config(), 1.2, 18)
    votes = CompactGroundingInspector._parse(j, envelope({"no_events": 0.95}))
    assert len(votes) == 2 and all(
        v.detected is False and not v.subject_ids for v in votes
    )
    with pytest.raises(_ResponseValidationError):
        CompactGroundingInspector._parse(
            replace(j, grounding_required=True), envelope({"no_events": 0.95})
        )


def test_followup_switch_tracks_pending_and_confirmed_then_grounded_clears():
    metadata, fusion, state = setup()
    metadata.observe_batch(batch(*(frame(i) for i in range(1, 6))))
    assert not fusion.grounding_required
    fusion.offer_vlm(vlm(state, 4))
    fusion.offer_vlm(vlm(state, 5))
    assert fusion.grounding_required
    metadata.observe_batch(batch(frame(6)))
    assert fusion.grounding_required
    fusion.offer_vlm(vlm(state, 6, False))
    fusion.offer_vlm(vlm(state, 7, False))
    metadata.observe_batch(batch(frame(7), frame(8)))
    assert not fusion.grounding_required


class OutputQueue(Queue):
    def close(self):
        pass

    def cancel_join_thread(self):
        pass


def test_default_mode_publishes_information_while_grounding_is_blocked():
    base = config()
    cfg = VLMProcessConfig(
        base.stream_session_id, base.source_id, base.generation, base.labels,
        base.model_settings, base.model, max_in_flight=base.max_in_flight,
    )
    incoming, outgoing = Queue(), OutputQueue()
    stop, release, started = Event(), Event(), Event()
    records = []

    class Inspector:
        def __init__(self, mode):
            self.mode = mode

        def inspect(self, j):
            records.append(
                (self.mode, j.request.request_id, tuple(f.jpeg_bytes for f in j.frames))
            )
            if self.mode == "compact":
                started.set()
                assert release.wait(4)
            return (verdict(),)

        def close(self):
            pass

    t = Thread(
        target=run_vlm_process,
        args=(cfg, incoming, outgoing, stop, lambda c: Inspector(c.output_mode)),
    )
    t.start()
    try:
        from goodvision_harness.scene_evidence import EvidenceFrame

        for seq, at in ((0, 0), (6, 0.4), (12, 0.8), (18, 1.2)):
            incoming.put(
                SourceSample(
                    EvidenceFrame(cfg.stream_session_id, seq, at, jpeg()),
                    utc_now(),
                    time.monotonic(),
                )
            )
        assert started.wait(2)
        reports = []
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            msg = outgoing.get(timeout=2)
            if msg["kind"] == "report":
                reports.append(msg["report"])
                break
        assert len(reports) == 1 and reports[0]["vlm_lane"] == "information"
        assert reports[0]["process_metrics"]["active_requests"] >= 1
        assert records[0][1:] == records[1][1:]
        release.set()
        while len(reports) < 2:
            msg = outgoing.get(timeout=2)
            if msg["kind"] == "report":
                reports.append(msg["report"])
        assert reports[1]["vlm_lane"] == "grounding"
        assert reports[0]["window_id"] == reports[1]["window_id"]
    finally:
        release.set()
        stop.set()
        t.join(5)
        assert not t.is_alive()


def test_information_and_grounding_have_separate_consumers():
    from types import SimpleNamespace

    from test_vlm_independent import await_condition, report

    from goodvision_harness.result_bus import ResultBus
    from goodvision_harness.vlm_independent import (
        IndependentEventState,
        IndependentVLMBranch,
    )

    cfg = replace(config(), output_mode="parallel")
    displayed, delivered, fused = [], [], []
    shared = ResultBus()
    display = SimpleNamespace(grounding_required=False, offer_vlm=displayed.append)
    fusion = SimpleNamespace(
        offer=lambda report, evidence: fused.append(report), close=lambda: None
    )
    branch = IndependentVLMBranch(
        cfg,
        results=shared,
        display_fusion=display,
        delivery=SimpleNamespace(submit=delivered.append),
        fusion=fusion,
    )
    branch._reader = Thread(target=branch._read)
    branch._reader.start()
    try:
        for lane in ("information", "grounding"):
            row = report(IndependentEventState(cfg), 1.2, 18, (verdict(),))
            row.update(vlm_lane=lane, window_id="paired-window")
            branch._outgoing.put({"kind": "report", "report": row})
        await_condition(lambda: len(fused) == 1)
        assert branch.results.latest().payload["vlm_lane"] == "information"
        assert shared.latest().payload["vlm_lane"] == "information"
        assert branch.grounding_results.latest().payload["vlm_lane"] == "grounding"
        assert [r["vlm_lane"] for r in displayed] == ["grounding"]
        assert [r["vlm_lane"] for r in delivered] == ["information"]
        assert [r["vlm_lane"] for r in fused] == ["grounding"]
    finally:
        branch.close(timeout_seconds=3)


@pytest.mark.parametrize("warmup_fails", [False, True])
def test_every_worker_prewarms_before_first_window_and_failure_is_fail_open(
    warmup_fails,
):
    from threading import Lock, get_ident

    from goodvision_harness.scene_evidence import EvidenceFrame

    cfg = replace(config(), output_mode="parallel")
    incoming, outgoing = Queue(), OutputQueue()
    stop, startup, release, all_warming = Event(), Event(), Event(), Event()
    warmed, inspected = [], []
    lock = Lock()
    slots = cfg.max_in_flight * 2

    class Inspector:
        def __init__(self, mode):
            self.mode, self.owner, self.warm_done = mode, get_ident(), False

        def warmup(self):
            assert get_ident() == self.owner
            with lock:
                warmed.append((self.mode, self.owner))
                if len(warmed) == slots:
                    all_warming.set()
            assert release.wait(3)
            self.warm_done = True
            if warmup_fails:
                raise RuntimeError("private transport detail")
            return True

        def inspect(self, inspection):
            assert get_ident() == self.owner and self.warm_done
            inspected.append((self.mode, self.owner))
            return (verdict(),)

        def close(self):
            pass

    worker = Thread(
        target=run_vlm_process,
        args=(
            cfg,
            incoming,
            outgoing,
            stop,
            lambda c: Inspector(c.output_mode),
            None,
            startup,
        ),
    )
    worker.start()
    try:
        assert all_warming.wait(3)
        assert len({owner for _, owner in warmed}) == slots
        assert not startup.is_set() and not inspected
        for seq, at in ((0, 0), (6, 0.4), (12, 0.8), (18, 1.2)):
            incoming.put(
                SourceSample(
                    EvidenceFrame(cfg.stream_session_id, seq, at, jpeg()),
                    utc_now(),
                    time.monotonic(),
                )
            )
        release.set()
        assert startup.wait(3)
        reports, status = [], None
        while len(reports) < 2:
            row = outgoing.get(timeout=3)
            if row["kind"] == "report":
                reports.append(row["report"])
            elif row.get("state") == "running":
                status = row
        assert status["stats"]["transport_warmup_slots"] == slots
        assert status["stats"]["transport_warmup_failed"] == (
            slots if warmup_fails else 0
        )
        assert all(r["state"] == "completed" for r in reports)
        assert {r["vlm_lane"] for r in reports} == {"information", "grounding"}
    finally:
        release.set()
        stop.set()
        worker.join(5)
        assert not worker.is_alive()


def test_information_deadline_cancels_body_and_keeps_client_usable(monkeypatch):
    import asyncio

    import httpx

    from goodvision_harness import scene_model, vlm_independent
    from goodvision_harness.model_api import ModelAPIError

    class SlowBody(httpx.AsyncByteStream):
        cancelled = False
        closed = False

        async def __aiter__(self):
            try:
                yield b" "
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def aclose(self):
            self.closed = True

    body, calls = SlowBody(), []

    async def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, stream=body)
        return httpx.Response(200, json=envelope({"smoking": False, "fall": False}))

    monkeypatch.setattr(vlm_independent, "INFORMATION_HTTP_DEADLINE_SECONDS", 0.05)
    monkeypatch.setattr(
        scene_model,
        "create_vlm_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    cfg = replace(config(), output_mode="events", api_key="TEST_KEY")
    inspector = vlm_independent._inspector(cfg)
    try:
        assert inspector.settings.timeout_seconds == 0.05
        with pytest.raises(ModelAPIError) as error:
            inspector.inspect(job(cfg, 1.2, 18))
        assert error.value.code == "timeout"
        assert body.cancelled and body.closed
        assert len(calls) == 1  # No retry that amplifies overload.
        assert all(not v.detected for v in inspector.inspect(job(cfg, 2.4, 36)))
        assert len(calls) == 2
    finally:
        inspector.close()
    shorter = vlm_independent._inspector(
        replace(
            cfg,
            model_settings=replace(cfg.model_settings, timeout_seconds=0.02),
        )
    )
    assert shorter.settings.timeout_seconds == 0.02
    shorter.close()
    grounded = vlm_independent._inspector(replace(cfg, output_mode="compact"))
    assert grounded.settings.timeout_seconds == cfg.model_settings.timeout_seconds
    grounded.close()


def test_fresh_information_bypasses_slow_rpc_and_full_grounding_pool():
    from threading import Lock

    from test_vlm_independent import await_condition

    from goodvision_harness.scene_evidence import EvidenceFrame

    cfg = replace(config(), output_mode="parallel")
    incoming, outgoing = Queue(), OutputQueue()
    stop, slow_started, release_slow, release_grounding = (Event() for _ in range(4))
    lock, active, maximum, inspected = Lock(), {}, {}, []

    class Inspector:
        def __init__(self, mode):
            self.mode = mode

        def inspect(self, j):
            endpoint = j.frames[-1].frame_sequence
            with lock:
                inspected.append((self.mode, endpoint))
                active[self.mode] = active.get(self.mode, 0) + 1
                maximum[self.mode] = max(maximum.get(self.mode, 0), active[self.mode])
            try:
                if self.mode == "compact":
                    assert release_grounding.wait(8)
                elif endpoint == 18:
                    slow_started.set()
                    assert release_slow.wait(8)
                # Explicitly test that a late positive cannot overwrite a newer
                # report; its discard must remain visible in counters.
                return (verdict(detected=endpoint == 18),)
            finally:
                with lock:
                    active[self.mode] -= 1

        def close(self):
            pass

    def send_window(end):
        # Seed the temporal history before starting the consumer. Later calls
        # advance it with one endpoint so an intermediate frame cannot be
        # dispatched while this producer is still constructing the window.
        sequences = range(0, 19, 6) if end == 18 else (end,)
        for seq in sequences:
            incoming.put(
                SourceSample(
                    EvidenceFrame(cfg.stream_session_id, seq, seq / 15, jpeg()),
                    utc_now(),
                    time.monotonic(),
                )
            )

    def receive_information():
        while True:
            item = outgoing.get(timeout=3)
            if (
                item["kind"] == "report"
                and item["report"].get("vlm_lane") == "information"
            ):
                return item["report"]

    worker = Thread(
        target=run_vlm_process,
        args=(
            cfg,
            incoming,
            outgoing,
            stop,
            lambda c: Inspector(c.output_mode),
        ),
    )
    send_window(18)
    worker.start()
    try:
        assert slow_started.wait(3)
        send_window(36)
        second = receive_information()
        assert second["frame"]["source_frame_index"] == 36
        assert second["timing"]["reorder_wait_seconds"] < 0.25
        await_condition(lambda: active.get("compact") == 2)
        send_window(54)
        third = receive_information()
        assert third["frame"]["source_frame_index"] == 54
        assert third["process_metrics"]["compact_busy_windows_skipped"] >= 1
        assert ("compact", 54) not in inspected
        assert not release_slow.is_set()
        release_slow.set()
        release_grounding.set()
        incoming.put(None)
        seen_information = []
        while True:
            row = outgoing.get(timeout=3)
            if (
                row["kind"] == "report"
                and row["report"].get("vlm_lane") == "information"
            ):
                seen_information.append(row["report"])
            if row.get("state") == "drained":
                assert row["stats"]["late_information_reports_discarded"] == 1
                assert row["stats"]["inflight_requests"] == 0
                break
        assert not seen_information
        assert all(n <= cfg.max_in_flight for n in maximum.values())
    finally:
        release_slow.set()
        release_grounding.set()
        stop.set()
        worker.join(8)
        assert not worker.is_alive()


def test_full_information_pool_admits_only_latest_window_after_capacity_returns():
    from test_vlm_independent import await_condition

    from goodvision_harness.scene_evidence import EvidenceFrame

    cfg = replace(config(), output_mode="events")
    incoming, outgoing = Queue(), OutputQueue()
    stop, release = Event(), Event()
    endpoints = []

    class Inspector:
        def inspect(self, j):
            endpoints.append(j.frames[-1].frame_sequence)
            assert release.wait(8)
            return (verdict(),)

        def close(self):
            pass

    def send(seq):
        incoming.put(
            SourceSample(
                EvidenceFrame(cfg.stream_session_id, seq, seq / 15, jpeg()),
                utc_now(),
                time.monotonic(),
            )
        )

    worker = Thread(
        target=run_vlm_process,
        args=(
            cfg,
            incoming,
            outgoing,
            stop,
            lambda _: Inspector(),
        ),
    )
    worker.start()
    try:
        send(0)
        send(18)
        await_condition(lambda: len(endpoints) == 1)
        send(36)
        await_condition(lambda: len(endpoints) == 2)
        for seq in (54, 72, 90):
            send(seq)
        await_condition(incoming.empty)
        assert endpoints == [18, 36]
        release.set()
        await_condition(lambda: len(endpoints) == 3)
        assert endpoints == [18, 36, 90]
    finally:
        release.set()
        stop.set()
        worker.join(8)
        assert not worker.is_alive()


def test_grounding_output_backpressure_keeps_both_lanes_progressing_and_retries_once():
    from goodvision_harness.scene_evidence import EvidenceFrame

    cfg = replace(config(), output_mode="parallel")
    incoming, stop, release = Queue(), Event(), Event()
    attempts = {}

    class GatedOutput(OutputQueue):
        def put_nowait(self, item):
            if (item["kind"] == "report"
                    and item["report"]["vlm_lane"] == "grounding"):
                report = item["report"]
                attempts.setdefault(report["request_id"], set()).add(
                    report["message_id"],
                )
                if not release.is_set():
                    raise Full
            super().put_nowait(item)

    class Inspector:
        def inspect(self, _job):
            return (verdict(),)

        def close(self):
            pass

    outgoing = GatedOutput()
    worker = Thread(target=run_vlm_process, args=(
        cfg, incoming, outgoing, stop, lambda _: Inspector(),
    ))
    worker.start()
    reports = []

    def send(sequence):
        incoming.put(SourceSample(
            EvidenceFrame(cfg.stream_session_id, sequence, sequence / 15, jpeg()),
            utc_now(), time.monotonic(),
        ))

    try:
        send(0)
        for sequence in (18, 36, 54, 72):
            send(sequence)
            while True:
                item = outgoing.get(timeout=5)
                if item["kind"] == "report":
                    report = item["report"]
                    assert report["vlm_lane"] == "information"
                    assert report["frame"]["source_frame_index"] == sequence
                    reports.append(report)
                    break
        # More grounding results are retained than there are inference slots;
        # completed grounding calls must not block either lane's next window.
        assert reports[-1]["process_metrics"]["lanes"]["compact"]["active_requests"] > 2
        incoming.put(None)
        release.set()
        while True:
            item = outgoing.get(timeout=5)
            if item["kind"] == "report":
                reports.append(item["report"])
            elif item.get("state") == "drained":
                assert item["stats"]["succeeded_requests"] == 8
                assert item["stats"]["active_requests"] == 0
                break
        assert len(reports) == len({r["message_id"] for r in reports}) == 8
        assert [r["sequence"] for r in reports] == list(range(1, 9))
        assert [r["frame"]["source_frame_index"] for r in reports
                if r["vlm_lane"] == "grounding"] == [18, 36, 54, 72]
        assert len(attempts) == 4 and all(len(ids) == 1 for ids in attempts.values())
    finally:
        release.set()
        stop.set()
        worker.join(8)
        assert not worker.is_alive()


@pytest.mark.parametrize("wrapper", [
    "{}", " \n{}\r\n ", "```json\n{}\n```",
    "```\n{}\n```", " \r\n```json \t\r\n\n{}\n\r\n``` \n",
])
@pytest.mark.parametrize("inspector,raw,expected", [
    (EventJSONInspector, {"smoking": True, "fall": None}, [True, None]),
    (CompactGroundingInspector, {"no_events": 0.9}, [False, False]),
    (CompactGroundingInspector, {"subjects": [], "verdicts": [
        {"event_label": "smoking", "detected": True},
        {"event_label": "fall", "detected": None},
    ]}, [True, None]),
])
def test_complete_reply_fence_keeps_event_semantics(wrapper, inspector, raw, expected):
    response = envelope(raw)
    response["choices"][0]["message"]["content"] = wrapper.format(json.dumps(raw))
    votes = inspector._parse(job(config(), 1.2, 18), response)
    assert [v.detected for v in votes] == expected


@pytest.mark.parametrize("content", [
    '```json\n{"smoking":false,"fall":false}\n``` extra',
    'prefix ```json\n{"smoking":false,"fall":false}\n```',
    '```json\n{"smoking":false,"fall":false}',
    '```python\n{"smoking":false,"fall":false}\n```',
    '```json\n{"smoking":false,"fall":false,}\n```',
    '```json\n{"smoking":"false","fall":false}\n```',
    '```json\n{"smoking":false}\n```',
    '```json\n{"smoking":false,"fall":false,"fall":true}\n```',
    '```json\n[]\n```',
])
def test_invalid_fenced_events_never_become_negative(content):
    response = envelope(None)
    response["choices"][0]["message"]["content"] = content
    with pytest.raises(_ResponseValidationError):
        EventJSONInspector._parse(job(config(), 1.2, 18), response)


@pytest.mark.parametrize("finish", [None, "length", "content_filter"])
def test_fence_does_not_hide_unfinished_completion(finish):
    response = envelope({"smoking": False, "fall": False})
    response["choices"][0]["finish_reason"] = finish
    with pytest.raises(_ResponseValidationError, match="completion_not_finished"):
        EventJSONInspector._parse(job(config(), 1.2, 18), response)
