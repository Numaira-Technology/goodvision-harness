"""Source-aligned camera protocol and small, independent display metadata bus."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import struct
import time
from collections import OrderedDict, deque
from threading import RLock

from .latency_trace import log_timing
from .privacy import public_safe_value
from .rich_labeling import validate_mask_contours
from .sam_display_fusion import display_subject_id


class CameraSocketChannel:
    """One websocket writer, with control/ACK ahead of independent display updates."""

    def __init__(self, websocket, metadata, *, interval=0.1):
        self.websocket, self.metadata = websocket, metadata
        self.interval = interval
        self.session_id = None
        self.cursor = None
        self.pending = None
        self.controls = deque()
        self.wake = asyncio.Event()
        self.closed = False
        self.writer = self.producer = None

    def start(self):
        self.writer = asyncio.create_task(self._write())
        self.producer = asyncio.create_task(self._produce())

    def enable(self, session_id):
        self.session_id, self.cursor, self.pending = session_id, None, None

    async def send(self, payload):
        if self.closed:
            raise RuntimeError("camera channel closed")
        delivered = asyncio.get_running_loop().create_future()
        self.controls.append((payload, delivered))
        self.wake.set()
        await delivered

    async def _produce(self):
        try:
            while not self.closed:
                if self.session_id is not None and self.pending is None:
                    session_id = self.session_id
                    payload = await asyncio.to_thread(
                        self.metadata.delta, self.cursor, stream_session_id=session_id,
                    )
                    if payload is not None and session_id == self.session_id:
                        self.pending = {"type": "camera_display_metadata", **payload}
                        self.wake.set()
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_timing("camera_display_transport_failure", stage="producer",
                       error_type=type(exc).__name__)
            self.closed = True
            await self.websocket.close(code=1011)

    @staticmethod
    def _encode(payload):
        text = json.dumps(
            payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"),
        )
        if len(text) > 1024 * 1024:
            raise ValueError("camera transport message capacity exceeded")
        return text

    async def flush_available(self):
        if self.session_id is None:
            return
        payload = await asyncio.to_thread(
            self.metadata.delta, self.cursor, stream_session_id=self.session_id,
        )
        if payload is not None and payload["stream_session_id"] == self.session_id:
            await self.send({"type": "camera_display_metadata", **payload})

    async def _write(self):
        delivered = None
        try:
            while not self.closed:
                await self.wake.wait()
                if self.controls:
                    payload, delivered = self.controls.popleft()
                    encoded = await asyncio.to_thread(self._encode, payload)
                    await asyncio.wait_for(self.websocket.send_text(encoded), timeout=5)
                    if (payload.get("type") == "camera_display_metadata"
                            and payload["stream_session_id"] == self.session_id):
                        self.cursor = max(
                            self.cursor or -1, payload["metadata_revision"],
                        )
                    if not delivered.done():
                        delivered.set_result(None)
                    delivered = None
                elif self.pending is not None:
                    payload = self.pending
                    if (payload["stream_session_id"] != self.session_id
                            or (self.cursor is not None
                                and payload["metadata_revision"] <= self.cursor)):
                        self.pending = None
                        continue
                    encoded = await asyncio.to_thread(self._encode, payload)
                    if self.controls:
                        continue  # ACKs arriving during encoding still go first.
                    if (payload is not self.pending
                            or payload["stream_session_id"] != self.session_id):
                        continue
                    started = time.monotonic()
                    await asyncio.wait_for(self.websocket.send_text(encoded), timeout=5)
                    log_timing("camera_display_delivery_timing",
                               stream_session_id=payload["stream_session_id"],
                               metadata_revision=payload["metadata_revision"],
                               anchors=len(payload["anchors"]), bytes=len(encoded),
                               send_seconds=time.monotonic() - started)
                    if payload["stream_session_id"] == self.session_id:
                        self.cursor = payload["metadata_revision"]
                    if self.pending is payload:
                        self.pending = None
                else:
                    self.wake.clear()
        except (Exception, asyncio.CancelledError) as exc:
            if not isinstance(exc, asyncio.CancelledError):
                log_timing("camera_display_transport_failure", stage="writer",
                           error_type=type(exc).__name__)
            self.closed = True
            if delivered is not None and not delivered.done():
                delivered.set_exception(RuntimeError("camera channel closed"))
            for _, waiter in self.controls:
                if not waiter.done():
                    waiter.set_exception(RuntimeError("camera channel closed"))
            self.controls.clear()
            try:
                await self.websocket.close(code=1011)
            except Exception:
                pass

    async def close(self):
        self.closed = True
        for task in (self.writer, self.producer):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (self.writer, self.producer)
                               if task is not None), return_exceptions=True)


def _source_time(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _confirmed_since(event: dict, scene: dict, previous: dict | None) -> float | None:
    value = event.get("confirmed_since_timestamp_seconds")
    if _source_time(value):
        return float(value)
    if "confirmed_since_timestamp_seconds" in event:
        return None
    candidates = [
        item.get("timestamp_seconds")
        for item in event.get("history", ())
        if item.get("phase") == "confirmed"
    ] + [
        item.get("timestamp_seconds")
        for item in scene.get("transitions", ())
        if item.get("event_id") == event.get("event_id")
        and item.get("phase") == "confirmed"
    ]
    if previous is not None and previous.get("phase") == "confirmed":
        candidates.append(previous["valid_from_timestamp"])
    usable = [value for value in candidates if _source_time(value)]
    return float(min(usable)) if usable else None


def decode_camera_packet(data: bytes) -> tuple[bytes, dict]:
    if data[:4] != b"GVC2":
        if len(data) <= 4:
            raise ValueError("camera frame is truncated")
        width, height = struct.unpack_from(">HH", data)
        return data[4:], {"width": width, "height": height}
    if len(data) < 8:
        raise ValueError("camera frame is truncated")
    size = struct.unpack_from(">I", data, 4)[0]
    if not 1 <= size <= 4096 or len(data) <= 8 + size:
        raise ValueError("invalid camera source header")
    try:
        header = json.loads(data[8 : 8 + size])
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid camera source header") from exc
    if not isinstance(header, dict) or set(header) != {
        "stream_session_id",
        "generation",
        "crop_version",
        "source_sequence",
        "media_timestamp",
        "width",
        "height",
    }:
        raise ValueError("invalid camera source fields")
    for key in ("generation", "crop_version", "source_sequence", "width", "height"):
        if type(header[key]) is not int or not 0 <= header[key] <= 2**53 - 1:
            raise ValueError("invalid camera source number")
    stamp = header["media_timestamp"]
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0:
        raise ValueError("invalid camera source timestamp")
    if not isinstance(header["stream_session_id"], str) or not (
        1 <= len(header["stream_session_id"]) <= 256
    ):
        raise ValueError("invalid camera run")
    return data[8 + size :], header


class CameraDisplayMetadata:
    """Only observed SAM anchors; estimated display geometry never enters here."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.reset(None)

    def reset(self, session_id: str | None, *, native_token=None,
              fusion=None) -> None:
        with self._lock:
            self.metadata_revision = getattr(self, "metadata_revision", -1) + 1
            self._reset_revision = self.metadata_revision
            self._native_token = native_token
            self._fusion = fusion
            self._native_epoch = None
            self._retired_epochs = set()
            self._anchor_revisions = {}
            self.session_id = session_id
            self.generation = self.crop_version = None
            self.last_source = (-1, -1.0)
            self.anchors = deque(maxlen=128)
            self.events = []
            self.identities = []
            self.revision = 0
            self.identity_revision = 0
            self._event_records: OrderedDict[tuple, dict] = OrderedDict()
            self._identity_records: OrderedDict[tuple, dict] = OrderedDict()
            self._active_identities: dict[tuple, tuple] = {}
            self._forgotten_identities: set[str] = set()
            self._scene_timestamp = -1.0

    def bind_native_run(self, session_id: str | None, native_token) -> bool:
        with self._lock:
            if session_id != self.session_id:
                return False
            self._native_token = native_token
            return True

    def source_identity(self) -> dict:
        with self._lock:
            return {"stream_session_id": self.session_id,
                    "generation": self.generation, "crop_version": self.crop_version}

    def accept_source(self, header: dict, *, commit: bool = True) -> bool:
        with self._lock:
            if header["stream_session_id"] != self.session_id:
                return False
            identity = (header["generation"], header["crop_version"])
            if self.generation is not None and identity != (
                self.generation,
                self.crop_version,
            ):
                return False
            source = (header["source_sequence"], header["media_timestamp"])
            if source[0] <= self.last_source[0] or source[1] <= self.last_source[1]:
                return False
            if commit:
                if self.generation is None:
                    self.metadata_revision += 1
                self.generation, self.crop_version = identity
                self.last_source = source
            return True

    def observe_batch(self, batch, *, native_token=None) -> bool:
        with self._lock:
            if (batch.stream_session_id != self.session_id
                    or (self._native_token is not None
                        and native_token != self._native_token)
                    or batch.predictor_epoch in self._retired_epochs):
                return False
            accepted = []
            for frame in batch.frames:
                prior = next((item for item in self.anchors
                              if item["source_sequence"] == frame.frame_sequence), None)
                if prior is not None:
                    # Validated native results are immutable; late L2 is idempotent.
                    continue
                if self.anchors and (
                    frame.frame_sequence <= self.anchors[-1]["source_sequence"]
                    or frame.timestamp_seconds <= self.anchors[-1]["media_timestamp"]
                ):
                    continue
                if self._native_epoch != batch.predictor_epoch:
                    if self._native_epoch is not None:
                        self._retired_epochs.add(self._native_epoch)
                    self._native_epoch = batch.predictor_epoch
                subjects, boxes = [], {}
                for item in frame.objects:
                    attrs = item.as_dict()["attributes"]
                    native = attrs.get("native_track", {})
                    contours = attrs.get("mask_contours")
                    if (
                        not isinstance(contours, dict)
                        or not native.get("object_id")
                        or native.get("epoch") != batch.predictor_epoch
                        or contours.get("source_frame_sequence") != frame.frame_sequence
                        or contours.get("source_timestamp_seconds")
                        != frame.timestamp_seconds
                        or (
                            item.observation_state
                            or attrs.get("observation_state", "observed")
                        )
                        != "observed"
                        or attrs.get("tracking_confirmation", "confirmed")
                        not in {"confirmed", "provisional"}
                    ):
                        continue
                    try:
                        validate_mask_contours(contours)
                    except (ValueError, TypeError):
                        continue
                    subjects.append(
                        {
                            "track_id": native["object_id"],
                            "polygons": copy.deepcopy(contours["polygons"]),
                            "confidence": item.confidence,
                            "tracking_confirmation": attrs.get(
                                "tracking_confirmation", "confirmed"
                            ),
                        }
                    )
                    if getattr(item, "bbox", None) is not None:
                        boxes[native["object_id"]] = item.bbox.as_dict()
                self.anchors.append(
                    {
                        "source_sequence": frame.frame_sequence,
                        "media_timestamp": frame.timestamp_seconds,
                        "epoch": batch.predictor_epoch,
                        "subjects": subjects,
                    }
                )
                accepted.append({**self.anchors[-1], "boxes": boxes})
                self.metadata_revision += 1
                self._anchor_revisions[frame.frame_sequence] = self.metadata_revision
                retained = {item["source_sequence"] for item in self.anchors}
                self._anchor_revisions = {
                    key: value for key, value in self._anchor_revisions.items()
                    if key in retained
                }
            if accepted and self._fusion is not None:
                previous_events = dict(self._event_records)
                previous_identities = dict(self._identity_records)
                try:
                    snapshot = self._fusion.advance(accepted)
                except Exception as exc:
                    # A semantic failure must still release this batch's masks.
                    log_timing("sam_display_association_failure",
                               error_type=type(exc).__name__)
                    return True
                self._apply_scene(snapshot)
                # Evidence keeps its original source time. Presentation changes
                # start on this SAM batch, never on an already published frame.
                boarding = accepted[-1]["media_timestamp"]
                for records, previous in (
                    (self._event_records, previous_events),
                    (self._identity_records, previous_identities),
                ):
                    for key, record in tuple(records.items()):
                        prior = previous.get(key)
                        same_phase = prior and prior.get("phase") == record.get("phase")
                        since = (prior["valid_from_timestamp"] if same_phase
                                 else max(boarding, record["valid_from_timestamp"]))
                        until = record["valid_until_timestamp"]
                        if until is not None:
                            until = (prior["valid_until_timestamp"]
                                     if same_phase and prior["valid_until_timestamp"]
                                     is not None else max(boarding, since, until))
                        records[key] = {**record, "valid_from_timestamp": since,
                                        "valid_until_timestamp": until}
                self.events = list(self._event_records.values())
                self.identities = list(self._identity_records.values())
            return True

    def observe_scene(self, snapshot: dict) -> None:
        with self._lock:
            if self._fusion is not None:
                self._fusion.offer_scene(snapshot)
                return
        self._apply_scene(snapshot)

    def _apply_scene(self, snapshot: dict) -> None:
        with self._lock:
            scene = snapshot.get("scene") or {}
            if scene.get("stream_session_id") != self.session_id:
                return
            now = scene.get("timestamp_seconds")
            if not _source_time(now) or now < self._scene_timestamp:
                return
            self._scene_timestamp = float(now)
            self.revision += 1
            self._observe_identities(scene, float(now))
            identities = {
                entity["subject_id"]: entity.get("native_track", {})
                for entity in scene.get("entities", ())
            }
            present = set()
            events = list(scene.get("events", ()))
            events.extend(
                {
                    "event_id": "identity:" + entity["subject_id"],
                    "event_label": "identity_pending",
                    "subject_ids": [entity["subject_id"]],
                    "phase": "suspected",
                    "first_evidence_timestamp_seconds": now,
                }
                for entity in scene.get("entities", ())
                if entity.get("identity_pending")
            )
            for event in events:
                for subject_id in event.get("subject_ids", ()):
                    native = identities.get(subject_id, {})
                    if not native.get("object_id") or not native.get("epoch"):
                        continue
                    key = (
                        event.get("event_id"),
                        native["epoch"],
                        native["object_id"],
                        event["event_label"],
                    )
                    if not isinstance(key[0], str) or not key[0]:
                        continue
                    phase = event.get("phase")
                    previous = self._event_records.get(key)
                    since = _confirmed_since(event, scene, previous)
                    display_phase = "confirmed"
                    if phase == "suspected" or (phase == "ended" and since is None):
                        display_phase = "suspected"
                        since = event.get("first_evidence_timestamp_seconds", now)
                        if (previous is not None and previous["phase"] == "suspected"
                                and previous["valid_until_timestamp"] is None):
                            since = previous["valid_from_timestamp"]
                        elif previous is not None:
                            since = now
                    if (phase not in {"suspected", "confirmed", "ended"}
                            or not _source_time(since)):
                        continue
                    until = None
                    if phase == "ended":
                        ends = [
                            item.get("timestamp_seconds")
                            for item in event.get("history", ())
                            if item.get("phase") == "ended"
                        ] + [
                            item.get("timestamp_seconds")
                            for item in scene.get("transitions", ())
                            if item.get("event_id") == key[0]
                            and item.get("phase") == "ended"
                        ]
                        explicit_end = event.get("ended_since_timestamp_seconds")
                        usable = (
                            [explicit_end]
                            if _source_time(explicit_end) and explicit_end >= since
                            else [
                                value
                                for value in ends
                                if _source_time(value) and value >= since
                            ]
                        )
                        fallback = event.get("last_evidence_timestamp_seconds")
                        until = min(usable) if usable else fallback
                        if not _source_time(until) or until < since:
                            until = max(now, since)
                    present.add(key)
                    self._event_records[key] = {
                        "event_id": key[0],
                        "epoch": native["epoch"],
                        "track_id": native["object_id"],
                        "event_label": event["event_label"],
                        "phase": display_phase,
                        "event_state": phase,
                        "valid_from_timestamp": since,
                        "valid_until_timestamp": until,
                        "revision": self.revision,
                    }
                    self._event_records.move_to_end(key)
            for key, prior in tuple(self._event_records.items()):
                if key not in present and prior["valid_until_timestamp"] is None:
                    self._event_records[key] = {
                        **prior,
                        "valid_until_timestamp": max(
                            now, prior["valid_from_timestamp"]
                        ),
                        "event_state": "withdrawn",
                        "revision": self.revision,
                    }
            while len(self._event_records) > 256:
                self._event_records.popitem(last=False)
            self.events = list(self._event_records.values())
            self.metadata_revision += 1

    def _observe_identities(self, scene: dict, now: float) -> None:
        """Name only an established gallery identity on its exact native track.

        Compact scene snapshots do not expose the face observation's source
        time. The scene watermark is a conservative effective time; body first
        seen/last seen timestamps must not backdate a newly learned identity.
        """
        known = {}
        for entity in scene.get("entities", ()):
            native = entity.get("native_track", {})
            identity = entity.get("identity", {})
            if not isinstance(native, dict) or not isinstance(identity, dict):
                continue
            epoch, track_id = native.get("epoch"), native.get("object_id")
            if not isinstance(epoch, str) or not epoch or not isinstance(track_id, str):
                continue
            if not track_id:
                continue
            key = (epoch, track_id)
            name, person_id = identity.get("name"), identity.get("person_id")
            value = None
            if (identity.get("status") == "known"
                    and isinstance(person_id, str) and person_id
                    and len(person_id) <= 256
                    and person_id not in self._forgotten_identities
                    and isinstance(name, str) and name.strip()
                    and len(name) <= 160
                    and not any(ord(char) < 32 or ord(char) == 127 for char in name)
                    and entity.get("tracking_confirmation", "confirmed")
                    == "confirmed"):
                value = {"person_id": person_id, "name": name.strip()}
            # Ambiguous scene aliases never choose a name by iteration order.
            if key in known and known[key] != value:
                known[key] = None
            else:
                known[key] = value
        known = {key: value for key, value in known.items() if value is not None}
        current = {
            key: {field: self._identity_records[record_key][field]
                  for field in ("person_id", "name")}
            for key, record_key in self._active_identities.items()
        }
        if known == current:
            return
        self.identity_revision += 1
        for key, record_key in tuple(self._active_identities.items()):
            prior = self._identity_records[record_key]
            value = known.get(key)
            if value is not None and all(prior[field] == value[field]
                                         for field in ("person_id", "name")):
                continue
            self._identity_records[record_key] = {
                **prior, "valid_until_timestamp": now,
                "revision": self.identity_revision,
            }
            del self._active_identities[key]
        for key, value in known.items():
            if value is None or key in self._active_identities:
                continue
            record_key = (*key, self.identity_revision)
            self._identity_records[record_key] = {
                "epoch": key[0], "track_id": key[1], **value,
                "valid_from_timestamp": now, "valid_until_timestamp": None,
                "revision": self.identity_revision,
            }
            self._active_identities[key] = record_key
        while len(self._identity_records) > 256:
            record_key, _ = self._identity_records.popitem(last=False)
            key = record_key[:2]
            if self._active_identities.get(key) == record_key:
                del self._active_identities[key]
        self.identities = list(self._identity_records.values())

    def forget_identity(self, person_id: str) -> None:
        """Gallery removal immediately revokes even buffered historical labels."""
        with self._lock:
            if self._fusion is not None:
                self._fusion.forget_identity(person_id)
            self._forgotten_identities.add(person_id)
            for record_key, record in tuple(self._identity_records.items()):
                if record["person_id"] == person_id:
                    del self._identity_records[record_key]
                    if self._active_identities.get(record_key[:2]) == record_key:
                        del self._active_identities[record_key[:2]]
            self.identities = list(self._identity_records.values())
            self.identity_revision += 1
            self.metadata_revision += 1

    def allow_identity(self, person_id: str) -> None:
        """Re-enrollment permits new scene evidence, without restoring old names."""
        with self._lock:
            if self._fusion is not None:
                self._fusion.allow_identity(person_id)
            self._forgotten_identities.discard(person_id)

    def display_snapshot(self, timestamp: float) -> dict:
        """Server rendering uses the same boarded semantics as browser contours."""
        with self._lock:
            anchor = next((item for item in reversed(self.anchors)
                           if item["media_timestamp"] <= timestamp), None)
            identities, events = self.identities, self.events
            now, session_id = self._scene_timestamp, self.session_id
        people, alerts = [], []
        if anchor is not None:
            for subject in anchor["subjects"]:
                epoch, track_id = anchor["epoch"], subject["track_id"]
                sid = display_subject_id(epoch, track_id)
                confirmed = subject["tracking_confirmation"] == "confirmed"
                name = next((item for item in reversed(identities)
                             if confirmed and item["epoch"] == epoch
                             and item["track_id"] == track_id
                             and item["valid_from_timestamp"] <= timestamp
                             and (item["valid_until_timestamp"] is None
                                  or timestamp < item["valid_until_timestamp"])), {})
                points = [p for polygon in subject["polygons"] for p in polygon]
                people.append({
                    "subject_id": sid, "entity_id": sid, "label": "person",
                    "bbox": {"x1": min(p[0] for p in points),
                             "y1": min(p[1] for p in points),
                             "x2": max(p[0] for p in points),
                             "y2": max(p[1] for p in points)},
                    "identity": {"status": "known", "name": name["name"],
                                 "person_id": name["person_id"]} if name else {},
                    "tracking_confirmation": subject["tracking_confirmation"],
                    "observation_state": "observed",
                    "last_observed_timestamp_seconds": anchor["media_timestamp"],
                    "frame_sequence": anchor["source_sequence"],
                    "mask_contours": {
                        "coordinate_system": "normalized",
                        "source_frame_sequence": anchor["source_sequence"],
                        "source_timestamp_seconds": anchor["media_timestamp"],
                        "polygons": subject["polygons"],
                    },
                })
                alerts.extend({
                    **event, "subject_ids": [sid],
                    "last_evidence_timestamp_seconds": event["valid_from_timestamp"],
                } for event in events if confirmed and event["epoch"] == epoch
                              and event["track_id"] == track_id
                              and event["valid_from_timestamp"] <= timestamp
                              and (event["valid_until_timestamp"] is None
                                   or timestamp < event["valid_until_timestamp"]))
        return {"stream_session_id": session_id, "entities": people,
                "events": alerts, "timestamp_seconds": timestamp,
                "event_state_timestamp_seconds": now, "state": "running"}

    def _snapshot_payload(self, after_revision=None):
        # Caller holds the lock. Published entries are immutable: writers append
        # or replace records/lists, never edit existing nested contour geometry.
        # Capture cheap references here; sanitize/copy outside the producer lock.
        newest = max(self.last_source[1],
                     self.anchors[-1]["media_timestamp"] if self.anchors else -1.0)
        return {
            "stream_session_id": self.session_id,
            "generation": self.generation, "crop_version": self.crop_version,
            "anchors": [anchor for anchor in self.anchors
                        if anchor["media_timestamp"] >= newest - 8.0
                        and (after_revision is None or
                             self._anchor_revisions.get(anchor["source_sequence"], -1)
                             > after_revision)],
            "events": self.events,
            "event_snapshot_timestamp": self._scene_timestamp,
            "event_snapshot_revision": self.revision,
            "identities": self.identities,
            "identity_snapshot_timestamp": self._scene_timestamp,
            "identity_snapshot_revision": self.identity_revision,
        }

    def delta(self, after_revision: int | None, *,
              stream_session_id: str) -> dict | None:
        with self._lock:
            if (stream_session_id != self.session_id or self.generation is None
                    or after_revision == self.metadata_revision):
                return None
            cursor = (after_revision if after_revision is not None
                      and after_revision >= self._reset_revision else None)
            payload = self._snapshot_payload(cursor)
            payload["metadata_revision"] = self.metadata_revision
        # Filter before copying. Repeated polls no longer traverse every old mask.
        return public_safe_value(payload)

    def snapshot(self) -> dict:
        with self._lock:
            payload = self._snapshot_payload()
        return public_safe_value(payload)
