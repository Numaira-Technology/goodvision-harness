"""Durable face admission, independent inference, one ordered identity commit."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from .face_observation_store import (
    FaceObservationStore,
    deserialize_objects,
    serialize_objects,
)
from .face_recognition import FaceGallerySnapshot, FaceObservation
from .grounding import GroundingResponse
from .latency_trace import log_timing, trace_id
from .privacy import PUBLIC_FACE_ERROR


class FaceObservationWorker:
    def __init__(self, owner, path, run_id):
        self.owner, self.run_id = owner, run_id
        self.store = FaceObservationStore(path, keep_open=True)
        self._condition = threading.Condition()
        self._closing = False
        self._submitted = self._completed = self._active = self._failed = 0
        self._last_source = (-1, -1.0)
        self._ready = {}
        self._futures = {}
        self._last_timing = {}
        self._executor = ThreadPoolExecutor(
            max_workers=owner.runtime.config.inference_instances,
            thread_name_prefix="face-stateless-inference",
        )
        self._committer = threading.Thread(
            target=self._commit_loop,
            name="face-ordered-identity",
            daemon=True,
        )
        self._committer.start()

    def submit(self, window, *, on_admitted=None):
        with self._condition:
            if self._closing:
                raise RuntimeError("face observation ingress is closed")
            source = window.newest_frame.source_frame_index, window.end_timestamp
            if source[0] <= self._last_source[0] or source[1] < self._last_source[1]:
                raise ValueError("face sources must be admitted in increasing order")
            ticket = self._submitted + 1
            gallery = self.owner.runtime.gallery.private_snapshot()
            self.store.admit(ticket, self.run_id, window, gallery.as_private_dict())
            future = Future()
            # Admission is durable: caller cancellation cannot silently remove
            # this source from the inference/identity/fact sequence.
            future.set_running_or_notify_cancel()
            future.face_ticket = ticket
            self._futures[ticket] = future
            self._submitted = ticket
            self._last_source = source
            # Delivery is registered before inference can finish. Its callback
            # executes on the committer before the next identity transition.
            try:
                if on_admitted is not None:
                    on_admitted(future)
                self._executor.submit(self._infer, ticket)
            except Exception as exc:
                error = type(exc).__name__
                self.store.fail(ticket, error)
                self._ready[ticket] = (error, time.monotonic(), time.monotonic())
                self._condition.notify_all()
            return future

    def _infer(self, ticket):
        error = None
        started = time.monotonic()
        with self._condition:
            self._active += 1
        try:
            window, gallery, _ = self.store.load(ticket)
            observation = self.owner.runtime.observe_window(
                window,
                gallery=FaceGallerySnapshot.from_private_dict(gallery),
            )
            self.store.record_raw(
                ticket,
                {
                    "schema_version": 1,
                    "objects": serialize_objects(observation.objects),
                    "started": observation.started,
                    "decode_seconds": observation.decode_seconds,
                    "inference_seconds": observation.inference_seconds,
                    "pool_wait_seconds": observation.pool_wait_seconds,
                    "error": observation.error,
                    "model_profile": {
                        "model_name": self.owner.runtime.config.model_name,
                        "threshold": self.owner.runtime.config.threshold,
                        "min_face_size": self.owner.runtime.config.min_face_size,
                    },
                },
            )
        except Exception as exc:
            error = type(exc).__name__
            try:
                self.store.fail(ticket, error)
            except Exception:
                error = "FaceObservationStorageFailure"
        finally:
            with self._condition:
                self._active -= 1
                self._ready[ticket] = (error, time.monotonic(), started)
                self._condition.notify_all()

    def _commit_loop(self):
        try:
            self._commit_pending()
        finally:
            # close(wait=False) still drains all admitted inference and ordered
            # callbacks. Release the connection only after that drain exits.
            self.store.close()

    def _commit_pending(self):
        while True:
            with self._condition:
                ticket = self._completed + 1
                self._condition.wait_for(
                    lambda ticket=ticket: (
                        ticket in self._ready
                        or (self._closing and self._completed == self._submitted)
                    )
                )
                if ticket not in self._ready:
                    return
                error, inferred_at, started = self._ready.pop(ticket)
                future = self._futures.pop(ticket)
            commit_started = time.monotonic()
            timing = {"ordered_commit_wait_seconds": commit_started - inferred_at}
            try:
                window, gallery_data, raw = self.store.load(ticket)
                if error is not None:
                    response = GroundingResponse(
                        window,
                        None,
                        time.monotonic() - started,
                        error=PUBLIC_FACE_ERROR,
                    )
                else:
                    observation = FaceObservation(
                        window,
                        deserialize_objects(raw["objects"]),
                        FaceGallerySnapshot.from_private_dict(gallery_data),
                        raw["started"],
                        raw["decode_seconds"],
                        raw["inference_seconds"],
                        raw["error"],
                        raw.get("pool_wait_seconds", 0.0),
                    )
                    timing.update(
                        decode_seconds=observation.decode_seconds,
                        inference_seconds=observation.inference_seconds,
                        pool_wait_seconds=observation.pool_wait_seconds,
                    )
                    commit_gallery = self.owner.runtime.gallery.private_snapshot()
                    response = self.owner.complete_observation(
                        observation,
                        gallery=commit_gallery,
                    )
                    classification = response.classification
                    online = {
                        "schema_version": 1,
                        "gallery_version": commit_gallery.version,
                        "objects": serialize_objects(
                            classification.face_observations if classification else ()
                        ),
                        "success": classification is not None,
                    }
                    if observation.error:
                        error = "FaceDecodeError"
                        self.store.fail(ticket, error)
                    else:
                        self.store.commit(ticket, online)
                timing["identity_commit_seconds"] = time.monotonic() - commit_started
                # set_result runs registered delivery callbacks before the next
                # ticket mutates identity state, preserving source order end to end.
                future.set_result(response)
            except Exception as exc:
                error = type(exc).__name__
                try:
                    self.store.fail(ticket, error)
                except Exception:
                    error = "FaceObservationStorageFailure"
                future.set_exception(RuntimeError("ordered face observation failed"))
            finally:
                with self._condition:
                    self._completed = ticket
                    self._failed += int(error is not None)
                    self._last_timing = timing
                    self._condition.notify_all()
                log_timing(
                    "face_pipeline_timing",
                    stream_session_id=trace_id(self.run_id),
                    ticket=ticket,
                    error_type=error,
                    outcome="error" if error else "ok",
                    instances=self.owner.runtime.config.inference_instances,
                    **timing,
                )

    def window_for(self, future):
        # Failed delivery still needs the admitted source/window identity.
        with self._condition:
            ticket = next(
                (key for key, value in self._futures.items() if value is future), None
            )
        if ticket is None:
            ticket = getattr(future, "face_ticket", None)
        if ticket is None:
            raise RuntimeError("face future has no admitted source")
        return self.store.load(ticket)[0]

    def stats(self):
        with self._condition:
            return {
                "scheduler": "durable_ordered_pool",
                "instances": self.owner.runtime.config.inference_instances,
                "inference_threads": self.owner.runtime.config.inference_threads,
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "active": self._active,
                "backlog": self._submitted - self._completed,
                **self._last_timing,
            }

    def close(self, *, wait=False):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._executor.shutdown(wait=wait, cancel_futures=False)
        if wait and threading.current_thread() is not self._committer:
            self._committer.join()
