"""Bounded predecode scheduling/ownership tests; no real GPU performance claim."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import replace

import httpx
import pytest
import test_sam_multistream as fixtures
from fastapi.testclient import TestClient
from PIL import Image
from test_sam31_continuous_worker import payload
from test_sam31_h264_sessions import codec as codec
from test_sam31_h264_sessions import wire

from workers.sam_scene.app import decode_jpeg
from workers.sam_scene.continuous import _RequestResources
from workers.sam_scene.multistream import FairAdmission


async def _until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


def _body(index, *, encoding="jpeg", count=1, start=0, final=False, size=(64, 40)):
    make = wire if encoding == "h264" and count else payload
    kwargs = {} if make is wire else {"size": size}
    result = json.loads(
        make(count, start=start, request_id=f"r-{start}", final=final, **kwargs)
    )
    result["stream_session_id"] = f"source-{index}"
    return result


async def _open(client, count):
    paths = []
    for index in range(1, count + 1):
        response = await client.post(
            fixtures._path(index), json={"stream_session_id": f"source-{index}"}
        )
        assert response.status_code == 200
        paths.append(
            f"{fixtures._path(index)}/{response.json()['video_session_id']}/append"
        )
    return paths


def _closed(images):
    for image in images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


@pytest.mark.parametrize("encoding", ["jpeg", "h264"])
def test_slow_decode_does_not_own_gpu_or_allow_lane_overtaking(
    monkeypatch, codec, encoding, caplog
):
    entered, release = threading.Event(), threading.Event()
    images = []

    def wait_first(result):
        images.extend(result)
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(
        fixtures,
        "decode_jpeg",
        lambda *args: wait_first([decode_jpeg(*args)])[0],
    )
    codec.callback = lambda data, **kw: wait_first(
        [Image.new("RGB", (64, 40)) for _ in range(kw["expected_frames"])]
    )
    app, fleet, gate, registry = fixtures._setup(2)
    for lane in fleet.engines.values():
        lane.settings = replace(lane.settings, h264_enabled=True)
    backend = fleet.source.backend
    original_run = backend.run

    def guarded_run(*args, **kwargs):
        assert gate.locked()
        yield from original_run(*args, **kwargs)

    backend.run = guarded_run

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://worker"
        ) as client:
            paths = await _open(client, 2)
            slow = asyncio.create_task(
                client.post(paths[0], json=_body(1, encoding=encoding))
            )
            try:
                await _until(entered.is_set)
                assert not gate.locked()
                assert fleet.engines["camera-01"].session.source_size is None
                assert backend.calls == 0
                busy = await client.post(paths[0], json=_body(1, start=1))
                assert busy.status_code == 429
                other = await asyncio.wait_for(
                    client.post(paths[1], json=_body(2, encoding=encoding)), 2
                )
                assert other.status_code == 200
                assert not slow.done()
                assert backend.calls == 1
            finally:
                release.set()
                result = await slow
            assert result.status_code == 200
            assert result.json()["accepted_frame_sequence"] == 0
            assert (await client.get("/health/cameras")).json()[
                "prepared_requests"
            ] == 0
            assert not gate.locked() and gate.waiting == 0
            _closed(images)

    try:
        with caplog.at_level("INFO", logger="goodvision.sam31.session"):
            asyncio.run(run())
        rows = [
            record.sam_timing
            for record in caplog.records
            if record.getMessage().startswith("Native session append timing")
        ]
        assert len(rows) == 2
        assert all(row["request_id"] == "r-0" for row in rows)
        assert all(row["decode_seconds"] == 0 for row in rows)
        assert all(row["preparation_decode_seconds"] > 0 for row in rows)
    finally:
        release.set()
        registry.close()


def test_prepared_buffers_are_bounded_even_while_gpu_is_busy(monkeypatch):
    images = []

    def decode(*args):
        result = decode_jpeg(*args)
        images.append(result)
        return result

    monkeypatch.setattr(fixtures, "decode_jpeg", decode)
    app, fleet, gate, registry = fixtures._setup(3)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://worker"
        ) as client:
            paths = await _open(client, 3)
            assert gate.acquire()
            tasks = []
            try:
                for index, path in enumerate(paths, 1):
                    tasks.append(
                        asyncio.create_task(client.post(path, json=_body(index)))
                    )
                    if index < 3:
                        await _until(lambda expected=index: gate.waiting == expected)
                await _until(lambda: len(images) == 2)

                # Third lane has reached the preparation queue, not a decoder.
                async def queued():
                    while True:
                        health = (await client.get("/health/cameras")).json()
                        if health["preparation_waiting_requests"] == 1:
                            return health
                        await asyncio.sleep(0.001)

                health = await asyncio.wait_for(queued(), 2)
                assert (
                    health["prepared_requests"] == health["max_prepared_requests"] == 2
                )
                assert len(images) == 2 and gate.waiting == 2
                tasks[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[0]
                await _until(lambda: len(images) == 3 and gate.waiting == 2)
                _closed(images[:1])
                assert gate.locked()
                assert fleet.engines["camera-01"].session.status == "active"
            finally:
                gate.release()
                results = await asyncio.gather(*tasks, return_exceptions=True)
            assert all(result.status_code == 200 for result in results[1:])
            assert not gate.locked() and gate.waiting == 0
            assert (await client.get("/health/cameras")).json()[
                "prepared_requests"
            ] == 0
            _closed(images)

    try:
        asyncio.run(run())
    finally:
        registry.close()


@pytest.mark.parametrize("encoding", ["jpeg", "h264"])
@pytest.mark.parametrize("failure", ["cancel", "timeout"])
def test_abandoned_decoder_keeps_budget_and_lane_until_actual_exit(
    monkeypatch, codec, failure, encoding
):
    entered, release = threading.Event(), threading.Event()
    images = []

    def wait_first(result):
        images.extend(result)
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(
        fixtures,
        "decode_jpeg",
        lambda *args: wait_first([decode_jpeg(*args)])[0],
    )
    codec.callback = lambda data, **kw: wait_first(
        [Image.new("RGB", (64, 40)) for _ in range(kw["expected_frames"])]
    )
    app, fleet, gate, registry = fixtures._setup(2)
    lane = fleet.engines["camera-01"]
    lane.settings = replace(lane.settings, h264_enabled=True)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://worker"
        ) as client:
            paths = await _open(client, 2)
            lane.video.settings = replace(
                lane.video.settings, request_timeout_seconds=0.2
            )
            request = asyncio.create_task(
                client.post(paths[0], json=_body(1, encoding=encoding))
            )
            try:
                await _until(entered.is_set)
                if failure == "cancel":
                    request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request
                else:
                    assert (await request).status_code == 504
                assert not gate.locked() and gate.waiting == 0
                assert lane.session.status == "active"
                assert (
                    lane.session.total_frames == 0 and lane.session.source_size is None
                )
                health = (await client.get("/health/cameras")).json()
                assert health["prepared_requests"] == 1
                assert (await client.post(paths[0], json=_body(1))).status_code == 429
                assert (await client.post(paths[1], json=_body(2))).status_code == 200
            finally:
                release.set()
                await asyncio.gather(request, return_exceptions=True)

            # Image closure alone precedes gate release; wait for a real retry.
            async def retry():
                while True:
                    result = await client.post(paths[0], json=_body(1))
                    if result.status_code != 429:
                        return result
                    await asyncio.sleep(0.001)

            assert (await asyncio.wait_for(retry(), 2)).status_code == 200
            assert lane.session.total_frames == 1
            assert (await client.get("/health/cameras")).json()[
                "prepared_requests"
            ] == 0
            _closed(images)

    try:
        asyncio.run(run())
    finally:
        release.set()
        registry.close()


@pytest.mark.parametrize("encoding", ["jpeg", "h264"])
def test_prepared_path_preserves_output_replay_eof_and_source_identity(
    monkeypatch, codec, encoding
):
    images = []

    def decode(*args):
        image = decode_jpeg(*args)
        images.append(image)
        return image

    monkeypatch.setattr(fixtures, "decode_jpeg", decode)
    app, fleet, _gate, registry = fixtures._setup(2)
    reference, prepared = fleet.engines.values()
    for lane in (reference, prepared):
        lane.settings = replace(lane.settings, h264_enabled=True)
    try:
        with TestClient(app) as client:
            tokens = [
                client.post(
                    fixtures._path(index), json={"stream_session_id": "source-1"}
                ).json()["video_session_id"]
                for index in (1, 2)
            ]
            prepared.session.predictor_epoch = reference.session.predictor_epoch
            for count, start, final in ((1, 0, False), (4, 1, False), (0, 5, True)):
                body = _body(
                    1, encoding=encoding, count=count, start=start, final=final
                )
                expected = reference.append(
                    tokens[0],
                    json.dumps(body).encode(),
                    deadline=time.monotonic() + 5,
                    decode_jpeg=decode_jpeg,
                )
                path = f"{fixtures._path(2)}/{tokens[1]}/append"
                actual = client.post(path, json=body)
                assert actual.status_code == 200
                expected["video_session_id"] = tokens[1]
                assert actual.json() == expected
                calls, decodes = (
                    fleet.source.backend.calls,
                    len(images) + len(codec.calls),
                )
                assert client.post(path, json=body).json() == expected
                assert fleet.source.backend.calls == calls
                assert len(images) + len(codec.calls) == decodes
            assert (
                prepared.session.status == "closed" and prepared.session.sources == {}
            )
            _closed(images + codec.images)
    finally:
        registry.close()


@pytest.mark.parametrize(
    "failure", ["decode", "capacity", "pixels", "resolution", "quota"]
)
def test_preparation_failure_cleanup_and_gpu_state_boundaries(monkeypatch, failure):
    images = []

    def decode(*args):
        image = decode_jpeg(*args)
        images.append(image)
        return image

    monkeypatch.setattr(fixtures, "decode_jpeg", decode)
    app, fleet, gate, registry = fixtures._setup(2)
    lane = fleet.engines["camera-01"]
    try:
        with TestClient(app) as client:
            token = client.post(
                fixtures._path(1), json={"stream_session_id": "source-1"}
            )
            path = f"{fixtures._path(1)}/{token.json()['video_session_id']}/append"
            body = _body(1)
            expected = 413
            if failure == "decode":
                body["frames"][0]["jpeg_base64"] = "YmFk"
                expected = 400
            elif failure == "capacity":
                body = _body(1, count=5)
            elif failure == "pixels":
                lane.video.settings = replace(
                    lane.video.settings, max_total_image_pixels=1
                )
            elif failure == "resolution":
                assert client.post(path, json=body).status_code == 200
                body = _body(1, start=1, size=(32, 20))
                expected = 409
            else:
                lane.session.total_frames = lane.settings.max_session_frames
                expected = 409
            calls = fleet.source.backend.calls
            result = client.post(path, json=body)
            assert result.status_code == expected
            assert fleet.source.backend.calls == calls
            assert lane.ready
            assert not gate.locked() and gate.waiting == 0
            assert client.get("/health/cameras").json()["prepared_requests"] == 0
            if failure in {"resolution", "quota"}:
                assert lane.session.status != "active"
                assert lane.session.native_state is None
            else:
                assert lane.session.status == "active"
                assert lane.session.source_size is None and lane.session.sources == {}
            if failure in {"capacity", "quota"}:
                assert not images
            _closed(images)
    finally:
        registry.close()


def test_cleanup_before_worker_dispatch_prevents_late_work_and_double_release():
    lane = threading.Lock()
    assert lane.acquire(False)
    preparation = FairAdmission(2, capacity=2)
    assert preparation.acquire()
    resources = _RequestResources(lane)
    resources.preparation = preparation
    resources.close()
    resources.close()
    with pytest.raises(TimeoutError):
        resources.run(lambda: pytest.fail("abandoned worker executed"))
    assert not lane.locked() and preparation.active == 0


@pytest.mark.parametrize("failure", ["cancel", "timeout"])
def test_preparation_queue_failure_leaves_session_and_other_leases_intact(
    monkeypatch, failure
):
    app, fleet, gpu, registry = fixtures._setup(3)
    original = FairAdmission.acquire_async
    waiting = threading.Event()
    preparation = []

    async def observe(self, deadline):
        if self.capacity == 2:
            preparation.append(self)
            if len(preparation) == 3:
                waiting.set()
        return await original(self, deadline)

    monkeypatch.setattr(FairAdmission, "acquire_async", observe)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://worker"
        ) as client:
            paths = await _open(client, 3)
            assert gpu.acquire()
            tasks = []
            try:
                for index in (1, 2):
                    tasks.append(
                        asyncio.create_task(
                            client.post(paths[index - 1], json=_body(index))
                        )
                    )
                    await _until(lambda expected=index: gpu.waiting == expected)
                # Per-lane video config object is shared; already dispatched
                # requests retain their original deadlines.
                lane = fleet.engines["camera-03"]
                lane.video.settings = replace(
                    lane.video.settings, request_timeout_seconds=0.1
                )
                third = asyncio.create_task(client.post(paths[2], json=_body(3)))
                tasks.append(third)
                await _until(waiting.is_set)
                if failure == "cancel":
                    third.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await third
                else:
                    assert (await third).status_code == 504
                assert preparation[0].active == 2 and preparation[0].waiting == 0
                assert gpu.locked() and gpu.waiting == 2
                assert (
                    lane.session.status == "active" and lane.session.total_frames == 0
                )
            finally:
                gpu.release()
                results = await asyncio.gather(*tasks, return_exceptions=True)
            assert all(result.status_code == 200 for result in results[:2])
            assert (await client.post(paths[2], json=_body(3))).status_code == 200
            assert preparation[0].active == 0 and preparation[0].waiting == 0

    try:
        asyncio.run(run())
    finally:
        registry.close()


def test_session_is_revalidated_after_preparation_and_expiry(monkeypatch):
    images = []

    def decode(*args):
        image = decode_jpeg(*args)
        images.append(image)
        return image

    monkeypatch.setattr(fixtures, "decode_jpeg", decode)
    app, fleet, gate, registry = fixtures._setup(2)
    lane = fleet.engines["camera-01"]

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://worker"
        ) as client:
            paths = await _open(client, 2)
            assert gate.acquire()
            request = asyncio.create_task(client.post(paths[0], json=_body(1)))
            try:
                await _until(lambda: gate.waiting == 1)
                lane.session.touched -= lane.settings.idle_seconds + 1
                # Like the idle reaper: native state cleanup owns the GPU slot.
                lane.expire()
                assert lane.session is None
            finally:
                gate.release()
                response = await request
            assert response.status_code == 404
            assert fleet.source.backend.calls == 0
            assert (await client.get("/health/cameras")).json()[
                "prepared_requests"
            ] == 0
            _closed(images)

    try:
        asyncio.run(run())
    finally:
        registry.close()


@pytest.mark.parametrize("after_grant", [False, True])
def test_two_slot_preparation_cancellation_preserves_other_owner(after_grant):
    async def run():
        gate = FairAdmission(2, capacity=2)
        assert gate.acquire() and gate.acquire()
        pending = asyncio.create_task(gate.acquire_async(time.monotonic() + 2))
        await _until(lambda: gate.waiting == 1)
        if after_grant:
            gate.release()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert gate.active == (1 if after_grant else 2)
        assert gate.waiting == 0
        while gate.active:
            gate.release()

    asyncio.run(run())
