"""Bounded camera lanes with one warmed SAM model and isolated temporal state."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

from fastapi.responses import JSONResponse

if __package__:
    from .continuous import Session, SessionEngine, register_session_routes
else:
    from continuous import Session, SessionEngine, register_session_routes


def camera_count_from_env():
    value = os.environ.get("SAM31_LIVE_CAMERAS", "1")
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError("SAM31_LIVE_CAMERAS must be a positive integer") from None
    if count < 1:
        raise ValueError("SAM31_LIVE_CAMERAS must be a positive integer")
    return count


@dataclass(eq=False)
class _Waiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    granted: bool = False


class FairGPUAdmission:
    """FIFO async waiters plus the existing nonblocking legacy lock interface.

    Worker threads release ownership only after actual model work finishes.
    Queue cancellation never releases somebody else's GPU operation.
    """

    def __init__(self, max_waiters):
        if type(max_waiters) is not int or max_waiters < 1:
            raise ValueError("max_waiters must be positive")
        self.max_waiters = max_waiters
        self._mutex = threading.RLock()
        self._active = False
        self._waiters = deque()

    def acquire(self, blocking=False):
        if blocking:
            raise ValueError("GPU admission only supports nonblocking sync callers")
        with self._mutex:
            if self._active or self._waiters:
                return False
            self._active = True
            return True

    def locked(self):
        with self._mutex:
            return self._active

    @property
    def waiting(self):
        with self._mutex:
            return len(self._waiters)

    async def acquire_async(self, deadline):
        if deadline <= time.monotonic():
            raise TimeoutError("GPU queue deadline exceeded")
        loop = asyncio.get_running_loop()
        waiter = _Waiter(loop, loop.create_future())
        with self._mutex:
            if not self._active and not self._waiters:
                self._active = True
                return True
            if len(self._waiters) >= self.max_waiters:
                return False
            self._waiters.append(waiter)
        try:
            await asyncio.wait_for(
                asyncio.shield(waiter.future), max(0, deadline - time.monotonic())
            )
            return True
        except BaseException:
            with self._mutex:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                granted = waiter.granted
                waiter.future.cancel()
            if granted:
                self.release()
            raise

    def release(self):
        with self._mutex:
            if not self._active:
                raise RuntimeError("GPU admission is not owned")
            self._active = False
            while self._waiters:
                waiter = self._waiters.popleft()
                if waiter.future.cancelled() or waiter.loop.is_closed():
                    continue
                waiter.granted = True
                self._active = True

                def grant(selected=waiter):
                    if not selected.future.done():
                        selected.future.set_result(True)

                waiter.loop.call_soon_threadsafe(grant)
                break


class _SharedBackend:
    """A lane can close itself without unloading another lane's model."""

    def __init__(self, source):
        self.source = source
        self._ready = False

    @property
    def gpu_ready(self):
        return self._ready and self.source.gpu_ready

    def warmup(self):
        if not self.source.gpu_ready:
            raise RuntimeError("shared native SAM model is unavailable")
        self._ready = True

    def close(self):
        self._ready = False

    def __getattr__(self, name):
        return getattr(self.source, name)


class _LaneOwnership:
    """Other camera lanes may coexist; every legacy route remains exclusive."""

    def __init__(self, fleet, engine):
        self.fleet = fleet
        self.engine = engine
        self.mutex = fleet.registry.mutex

    @property
    def has_active_session(self):
        return self.engine.has_active_session or any(
            engine.has_active_session
            for engine in self.fleet.registry.engines
            if engine is not self.fleet
        )

    def expire(self):
        self.fleet.registry.expire()

    def note_state_discarded(self):
        self.fleet.registry.note_state_discarded()


class CameraFleet:
    """Aggregate ownership for legacy exclusion; one SessionEngine per camera."""

    def __init__(self, source, registry, count):
        if type(count) is not int or count < 2:
            raise ValueError("camera fleet must have at least 2 lanes")
        self.source = source
        self.registry = registry
        self.engines = {}
        for index in range(1, count + 1):
            engine = SessionEngine(
                source.video,
                source.settings,
                _SharedBackend(source.backend),
                profile=source.profile,
                production=source._production,
            )
            engine.registry = _LaneOwnership(self, engine)
            self.engines[f"camera-{index:02d}"] = engine
        # Register an aggregate watcher, not individual lanes: SessionRegistry.add
        # would otherwise overwrite each lane's separate ownership domain.
        registry.add(self)

    @property
    def has_active_session(self):
        return any(engine.has_active_session for engine in self.engines.values())

    def expire(self):
        for engine in self.engines.values():
            engine.expire()

    def close(self):
        for engine in self.engines.values():
            engine.close()

    def warmup(self):
        try:
            if not self.source.ready:
                raise RuntimeError("shared SAM route is unavailable")
            if self.source._production:
                self._verify_interleaved_states()
            for engine in self.engines.values():
                engine.warmup()
        except Exception:
            for engine in self.engines.values():
                engine.error_code = "camera_initialization_failed"
            self.close()
            raise

    def _verify_interleaved_states(self):
        """Exercise ring rollover on two real CUDA states before advertising lanes."""
        from PIL import Image

        backend = self.source.backend
        if not backend.gpu_ready:
            raise RuntimeError("shared native SAM model is unavailable")
        sessions = [
            Session(f"lane-probe-{i}", f"lane-probe-{i}", "warmup", time.monotonic())
            for i in range(2)
        ]
        try:
            for count in (1, 4, 4, 4, 4, 4, 4, 4, 4):
                for index, session in enumerate(sessions):
                    size = (320, 192) if index == 0 else (256, 160)
                    images = [
                        Image.new("RGB", size, (index * 80, 0, 0)) for _ in range(count)
                    ]
                    try:
                        outputs = list(backend.run(session, images, final=False))
                        expected = list(
                            range(session.total_frames, session.total_frames + count)
                        )
                        if [frame for frame, _ in outputs] != expected:
                            raise RuntimeError(
                                "interleaved SAM omitted source frames"
                            )
                        session.total_frames += count
                        session.emitted_index = expected[-1]
                        state = session.native_state
                        if (
                            state["num_frames"] != session.total_frames
                            or (state["orig_width"], state["orig_height"]) != size
                        ):
                            raise RuntimeError("interleaved SAM changed source state")
                        backend.prune(session)
                    finally:
                        for image in images:
                            image.close()
                if sessions[0].native_state is sessions[1].native_state:
                    raise RuntimeError("interleaved SAM shared temporal state")
                for key in (
                    "input_batch", "generator_state", "tracker_metadata",
                    "feature_cache",
                ):
                    if sessions[0].native_state[key] is sessions[1].native_state[key]:
                        raise RuntimeError("interleaved SAM shared temporal storage")
            for session in sessions:
                if list(backend.run(session, [], final=True)):
                    raise RuntimeError("interleaved SAM invented EOF frames")
        finally:
            for session in sessions:
                backend.discard(session)


def register_camera_routes(app, fleet, admission, decode_jpeg):
    app.state.camera_fleet = fleet
    for camera_id, engine in fleet.engines.items():
        register_session_routes(
            app,
            engine,
            admission,
            decode_jpeg,
            api_prefix=f"/v1/cameras/{camera_id}/perception/sam_live",
            health_path=f"/health/cameras/{camera_id}/perception/sam_live",
            legacy=False,
            queued_admission=True,
        )

    @app.get("/health/cameras")
    async def health():
        ready = all(engine.gpu_ready for engine in fleet.engines.values())
        return JSONResponse(
            {
                "enabled": fleet.source.settings.enabled,
                "ready": ready,
                "gpu_ready": ready,
                "model_instances": 1,
                "max_active_sessions": len(fleet.engines),
                "active_sessions": sum(
                    engine.has_active_session for engine in fleet.engines.values()
                ),
                "max_gpu_operations": 1,
                "queued_requests": admission.waiting,
                "max_queued_requests": admission.max_waiters,
                "cameras": [
                    {
                        "id": camera_id,
                        "ready": engine.gpu_ready,
                        "active_session": engine.has_active_session,
                        "api_prefix": f"/v1/cameras/{camera_id}/perception/sam_live",
                        "health_path": (
                            f"/health/cameras/{camera_id}/perception/sam_live"
                        ),
                    }
                    for camera_id, engine in fleet.engines.items()
                ],
            },
            status_code=200 if ready else 503,
        )
