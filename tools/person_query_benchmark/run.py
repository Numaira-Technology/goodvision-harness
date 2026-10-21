"""Record paced person queries from explicit videos and an existing detector service.

Example: python -m tools.person_query_benchmark.run --execute --video id=/path/video.mp4
--queries /path/public-queries.jsonl --endpoint http://127.0.0.1:8091/v1/tracks
--output /path/new-recording. Repeated --video entries run sequentially.
"""

import argparse
import base64
import json
import math
from pathlib import Path
from threading import Event
import time
from urllib.parse import urlparse

import httpx

from goodvision_harness.rich_labeling import RichLabelCoordinator, RichLabelHTTPClient
from goodvision_harness.video_to_stream import VideoFileStream, encode_jpeg
from tools.videojev_benchmark.recording import EventStreamRecorder, publish_once

from .publishing import PersonQueryPublisher

_QUERY_FIELDS = {"query_id", "video_id", "task", "query_time_seconds",
                 "query_frame_index", "question"}


class _ResponseTransport(httpx.BaseTransport):
    """Retain the response body when the existing client rejects a response."""

    def __init__(self):
        self.inner, self.last_response = httpx.HTTPTransport(), None

    def handle_request(self, request):
        response = self.inner.handle_request(request)
        raw = response.read()
        try:
            body = {"response_body": raw.decode("utf-8")}
        except UnicodeDecodeError:
            body = {"response_body_base64": base64.b64encode(raw).decode("ascii")}
        self.last_response = {"status_code": response.status_code, **body}
        return response

    def close(self):
        self.inner.close()


class _RecordingClient:
    def __init__(self, args, recorder, stop):
        self.recorder, self.stop = recorder, stop
        self.cooldown_until = 0.0
        self.shared_error = None
        self.transport = _ResponseTransport()
        headers = ({"x-goodvision-detector-model": args.detector_model}
                   if args.detector_model else None)
        self.client = RichLabelHTTPClient(
            args.endpoint, timeout_seconds=args.http_timeout_seconds,
            mode="tracks", include_attributes=False, headers=headers,
            transport=self.transport,
        )

    def infer(self, jpeg_bytes, **metadata):
        if self.shared_error is not None:
            try:
                self.recorder.failure("person_request_not_sent", **metadata,
                                      shared_service_error=self.shared_error)
            except Exception:
                self.stop.set()
            raise RuntimeError(self.shared_error)
        self.transport.last_response = None
        try:
            return self.client.infer(jpeg_bytes, **metadata)
        except Exception as error:
            cooldown = 0.0
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in {401, 403}:
                self.shared_error = "http_" + str(error.response.status_code)
                self.stop.set()
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429:
                try:
                    retry_after = float(error.response.headers.get("Retry-After", "60"))
                except ValueError:
                    retry_after = 60.0
                cooldown = max(60.0, retry_after if math.isfinite(retry_after) else 60.0)
                self.cooldown_until = time.monotonic() + cooldown
            try:
                self.recorder.failure("person_request_failed", **metadata,
                                      error_type=type(error).__name__, error=str(error),
                                      cooldown_seconds=cooldown,
                                      **(self.transport.last_response or {}))
            except Exception:
                self.stop.set()
            if cooldown:
                while time.monotonic() < self.cooldown_until:
                    time.sleep(min(0.5, max(0, self.cooldown_until - time.monotonic())))
            raise

    def close(self):
        self.client.close()


def _read_queries(paths, video_ids):
    queries, seen = [], set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            query = json.loads(line)
            if not isinstance(query, dict) or set(query) != _QUERY_FIELDS:
                raise ValueError("provide only public query fields; references are not inputs")
            if (any(not isinstance(query[key], str) or not query[key]
                    for key in ("query_id", "video_id", "task", "question"))
                    or type(query["query_frame_index"]) is not int
                    or query["query_frame_index"] < 0
                    or type(query["query_time_seconds"]) not in (int, float)
                    or not math.isfinite(query["query_time_seconds"])
                    or query["query_time_seconds"] < 0
                    or query["query_id"] in seen):
                raise ValueError("invalid or duplicated public query")
            seen.add(query["query_id"])
            if query["video_id"] in video_ids:
                queries.append(query)
    return queries


class _SourceRun:
    def __init__(self, args, video_id, video, directory, queries):
        self.args, self.video_id, self.video = args, video_id, video
        self.stop, self.error = Event(), None
        self.recording_failed = False
        self.stream = self.coordinator = self.recorder = self.query_file = None
        self.client = None
        self.publisher = None
        self.queries = queries
        self.pending_queries = {}
        for query in queries:
            self.pending_queries.setdefault(query["query_frame_index"], []).append(query)
        self.input_map, self.submitted = {}, {}
        self.directory = directory
        self.session_id = "person-query:" + directory.parent.name + ":" + video_id
        self.range_end = False
        self.last_raw_index = None
        self.encoded = self.admitted = 0

    def fail(self, reason, error):
        if self.error is None:
            self.error = reason + ":" + type(error).__name__
        self.stop.set()
        if self.stream is not None:
            self.stream.stop()
        if self.recorder is not None and not self.recorder.failed:
            try:
                self.recorder.failure(reason, error_type=type(error).__name__, error=str(error))
            except Exception:
                self.recording_failed = True
                self.error = "recording_failed"

    def write_query(self, row):
        try:
            self.query_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self.query_file.flush()
        except Exception as error:
            self.recording_failed = True
            self.fail("query_recording_failed", error)
            raise

    def source_input(self, row):
        if (self.args.end_seconds is not None
                and row["timestamp_seconds"] >= self.args.end_seconds):
            self.range_end = True
            self.stream.stop()
            return
        try:
            self.recorder.source_input(row)
            self.last_raw_index = row["raw_frame_index"]
            if row["frame_sequence"] is not None:
                self.input_map[row["frame_sequence"]] = (
                    row["raw_frame_index"], row["timestamp_seconds"])
            for query in self.pending_queries.pop(row["raw_frame_index"], []):
                self.write_query(self.publisher.read_query(query, row["ingress_monotonic"]))
        except Exception as error:
            self.fail("input_recording_failed", error)
            raise

    def snapshot(self, snapshot):
        try:
            original, size = self.submitted[snapshot.frame_sequence]
            self.publisher.publish(snapshot, source_frame_index=original, source_size=size)
        except Exception as error:
            # RichLabelCoordinator contains callback exceptions; stop the source explicitly.
            self.fail("person_publication_failed", error)
            raise

    def run(self):
        eof = drained = False
        summary = {}
        try:
            self.recorder = EventStreamRecorder(self.directory, {
                "video_id": self.video_id, "video": str(self.video),
                "stream_session_id": self.session_id,
                "clock": "parent_process_time.monotonic", "endpoint": self.args.endpoint,
                "requested_detector_model": self.args.detector_model,
                "sample_fps": self.args.sample_fps, "max_image_side": self.args.max_image_side,
                "jpeg_quality": self.args.jpeg_quality,
                "http_timeout_seconds": self.args.http_timeout_seconds,
                "drain_timeout_seconds": self.args.drain_timeout_seconds,
                "start_seconds": self.args.start_seconds, "end_seconds": self.args.end_seconds,
                "ground_truth_input": False,
            })
            self.query_file = (self.directory / "queries.jsonl").open("x", encoding="utf-8")
            self.publisher = PersonQueryPublisher(self.video_id, self.recorder)
            self.stream = VideoFileStream(
                self.video, display_fps=self.args.sample_fps, start_seconds=self.args.start_seconds,
                input_observer=self.source_input,
            )
            self.client = _RecordingClient(self.args, self.recorder, self.stop)
            self.coordinator = RichLabelCoordinator(
                self.client,
                max_fps=self.args.sample_fps, history_size=128, on_snapshot=self.snapshot,
            )
            session_id = self.session_id
            self.coordinator.reset(session_id)
            iterator = self.stream.frames()
            try:
                for frame in iterator:
                    if self.stop.is_set() or self.range_end:
                        break
                    original, timestamp = self.input_map.pop(frame.index)
                    height, width = frame.bgr.shape[:2]
                    jpeg, _, _ = encode_jpeg(
                        frame.bgr, max_image_side=self.args.max_image_side,
                        quality=self.args.jpeg_quality,
                    )
                    self.encoded += 1
                    self.submitted[frame.index] = (original, {"width": width, "height": height})
                    # Source cadence selects candidates; busy work keeps the latest pending one.
                    self.admitted += int(self.coordinator.submit(
                        jpeg, stream_session_id=session_id, frame_sequence=frame.index,
                        timestamp_seconds=timestamp,
                    ))
                else:
                    eof = (not self.stream.stopped and self.last_raw_index is not None
                           and self.last_raw_index >= self.stream.info.frame_count - 1)
                    if not self.stream.stopped and not eof:
                        raise RuntimeError("decoder ended before the reported source frame count")
            finally:
                iterator.close()
            deadline = time.monotonic() + self.args.drain_timeout_seconds
            while True:
                deadline = max(deadline, self.client.cooldown_until + self.args.drain_timeout_seconds)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if self.coordinator.wait_until_idle(min(remaining, 0.5)):
                    drained = True
                    break
            if not drained:
                self.fail("person_drain_timeout", TimeoutError("person requests did not drain"))
        except Exception as error:
            self.fail("source_failed", error)
        finally:
            if self.coordinator is not None:
                try:
                    self.coordinator.close(timeout_seconds=self.args.http_timeout_seconds + 1)
                except Exception as error:
                    self.fail("person_close_failed", error)
                if not self.coordinator.close_complete():
                    self.fail("person_worker_still_closing", RuntimeError("person worker has not retired"))
            elif self.client is not None:
                try:
                    self.client.close()
                except Exception as error:
                    self.fail("person_close_failed", error)
            if self.query_file is not None:
                for queries in self.pending_queries.values():
                    for query in queries:
                        try:
                            self.write_query({**query, "ingress_monotonic": None,
                                              "read_monotonic_lower": None,
                                              "read_monotonic_upper": None,
                                              "selected_report_reference": None,
                                              "newer_publication_boundary_uncertain": False,
                                              "unknown_reason": "query_frame_not_ingested"})
                        except Exception:
                            break
                try:
                    self.query_file.close()
                except Exception as error:
                    self.recording_failed = True
                    self.fail("query_recording_close_failed", error)
            source_complete = eof or self.range_end
            if (eof and self.args.end_seconds is not None and self.stream is not None
                    and self.args.end_seconds > self.stream.info.duration_seconds + 1e-9):
                source_complete = False
                if self.error is None:
                    self.error = "media_ended_before_requested_range"
            shared_error = self.client.shared_error if self.client is not None else None
            if shared_error is not None and self.error is None:
                self.error = "shared_service_error:" + shared_error
            summary = {"video_id": self.video_id, "original_eof": eof,
                       "range_end_reached": self.range_end, "source_complete": source_complete,
                       "drained": drained, "error": self.error,
                       "shared_service_error": shared_error,
                       "encoded": self.encoded, "admitted": self.admitted,
                       "worker_closed": self.coordinator is None or self.coordinator.close_complete(),
                       "recording_failed": self.recording_failed or bool(self.recorder and self.recorder.failed),
                       "tracking": self.coordinator.stats() if self.coordinator else {},
                       "input_timing": self.stream.input_timing_statistics if self.stream else {}}
            if self.recorder is not None:
                try:
                    self.recorder.finish(summary)
                except Exception as error:
                    self.recording_failed = True
                    self.fail("completion_recording_failed", error)
                    summary["error"] = self.error
                    summary["recording_failed"] = True
                try:
                    self.recorder.close()
                except Exception as error:
                    summary["error"] = "recording_close_failed:" + type(error).__name__
                    summary["recording_failed"] = True
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly allow video and detector requests")
    parser.add_argument("--video", action="append", required=True, metavar="VIDEO_ID=PATH")
    parser.add_argument("--queries", action="append", required=True, help="Public query JSONL; no references")
    parser.add_argument("--endpoint", required=True, help="Existing detector /v1/tracks or /v1/labels endpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector-model", choices=("yolov8-m", "yolo26-s", "rfdetr-l"))
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--http-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--drain-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--max-image-side", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required; no video, service, or output was opened")
    for key in ("sample_fps", "http_timeout_seconds", "drain_timeout_seconds"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(key + " must be finite and positive")
    if (not math.isfinite(args.start_seconds) or args.start_seconds < 0
            or (args.end_seconds is not None and (not math.isfinite(args.end_seconds)
                or args.end_seconds <= args.start_seconds))
            or not 1 <= args.jpeg_quality <= 100
            or (args.max_image_side is not None and args.max_image_side <= 0)):
        parser.error("invalid source range or JPEG settings")
    endpoint = urlparse(args.endpoint)
    if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
            or endpoint.path not in {"/v1/tracks", "/v1/labels"}):
        parser.error("provide an explicit detector endpoint without credentials")
    sources = []
    for value in args.video:
        video_id, separator, path = value.partition("=")
        if not separator or not video_id or not path or video_id in {item[0] for item in sources}:
            parser.error("provide distinct VIDEO_ID=PATH entries")
        sources.append((video_id, Path(path).expanduser().resolve()))
    queries = _read_queries(args.queries, {item[0] for item in sources})
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, (video_id, video) in enumerate(sources, 1):
        result = _SourceRun(args, video_id, video, output / f"{index:04d}",
                            [query for query in queries if query["video_id"] == video_id]).run()
        results.append(result)
        if (result["recording_failed"] or not result["worker_closed"]
                or result["shared_service_error"] is not None):
            break
    publish_once(output / "run.json", {"sources": results})
    print(json.dumps({"sources": results}, ensure_ascii=False, allow_nan=False))
    return 0 if all(row["source_complete"] and row["drained"] and row["error"] is None
                    for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
