"""Real browser viewport recording, finalized before it is marked deliverable."""

from __future__ import annotations

import concurrent.futures
import json
import math
import queue
import threading
import time
from fractions import Fraction
from pathlib import Path

from tools.vlm_benchmark.report import write_json

_METADATA_LOCK = threading.Lock()


def save_recording(path, record):
    # The worker and interrupted main thread can finalize the same artifact.
    # Serialize writes because write_json uses one temporary path per file.
    with _METADATA_LOCK:
        write_json(path, record)


def transcode(source, destination, *, preview=False):
    import av

    fps = 15 if preview else 25
    count, end = 0, 0.0
    with (
        av.open(str(source)) as reader,
        av.open(str(destination), "w", options={"movflags": "+faststart"}) as writer,
    ):
        stream = reader.streams.video[0]
        width, height = stream.width, stream.height
        if preview:
            scale = min(1, 720 / max(width, height))
            width, height = int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
        out = writer.add_stream("libx264", rate=fps)
        out.width, out.height, out.pix_fmt = width, height, "yuv420p"
        # Keep variable-rate source timestamps precise. The encoder's default
        # 1/fps timebase can round two nearby timestamps to the same packet PTS.
        out.time_base = out.codec_context.time_base = Fraction(1, 90000)
        out.options = {"preset": "veryfast", "crf": "23"}
        out.codec_context.thread_count = 2
        origin = float((stream.start_time or 0) * stream.time_base)
        due = 0.0
        for frame in reader.decode(video=0):
            if frame.time is None:
                raise RuntimeError("video frame has no timestamp")
            at = max(0.0, frame.time - origin)
            if preview and at + 1e-6 < due:
                continue
            # Advance a fixed media clock, rather than accumulating rounding
            # to the next input frame (which undersamples e.g. 50 -> 15 FPS).
            due = (math.floor((at + 1e-6) * fps) + 1) / fps
            converted = frame.reformat(width=width, height=height, format="yuv420p")
            converted.time_base = Fraction(1, 90000)
            converted.pts = round(at * 90000)
            for packet in out.encode(converted):
                writer.mux(packet)
            count += 1
            end = at + 1 / fps
        for packet in out.encode():
            writer.mux(packet)
    if not count:
        raise RuntimeError("video contains no frames")
    return {"frames": count, "duration_seconds": end, "width": width, "height": height}


def make_preview(source, destination):
    return transcode(source, destination, preview=True)


def inspect_video(path):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first, last, frames = None, None, 0
        for frame in container.decode(video=0):
            if frame.time is None or (last is not None and frame.time <= last):
                raise RuntimeError("recording timestamps are missing or nonmonotonic")
            first = frame.time if first is None else first
            last, frames = frame.time, frames + 1
        if not frames:
            raise RuntimeError("recording is empty")
        return {
            "frames": frames,
            "duration_seconds": last - first + 1 / float(stream.average_rate or 25),
            "width": stream.width,
            "height": stream.height,
        }


def recording_is_complete(
    measured, elapsed, diagnostics, errors, *, benchmark_completed=True
):
    """A finalized file alone does not prove the live source was visible."""
    return bool(
        benchmark_completed
        and measured["duration_seconds"] >= elapsed - 1.0
        and diagnostics["sawPlaying"]
        and diagnostics["finalPainted"]
        and diagnostics.get("pendingEvidence") == 0
        and not errors
        and not diagnostics["mediaErrors"]
        and diagnostics["maxPollGapSeconds"] < 2.0
        and diagnostics["maxPlaybackGapSeconds"] < 2.0
        and all(n is None or n > 1 for n in diagnostics["videoFrames"])
    )


class BrowserRecorder:
    """Keep Playwright on its own thread, never on the timed frame-sending loop."""

    def __init__(self, url, headed=False):
        self.url, self.headed = url, headed
        self.commands = queue.Queue()
        self.ready = concurrent.futures.Future()
        self.cancelled = threading.Event()
        self.folder = None
        self.failure = None
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        self.ready.result(timeout=60)

    def call(self, name, *args):
        self.check()
        future = concurrent.futures.Future()
        self.commands.put((name, args, future))
        try:
            return future.result(timeout=180)
        except BaseException:
            # A main-thread interrupt can arrive while the worker is encoding.
            # Do not let its late completion relabel an aborted capture complete.
            self.cancelled.set()
            raise

    def check(self):
        if self.failure:
            raise RuntimeError(
                "browser recording failed; inspect recording.json"
            ) from self.failure

    def begin(self, location, stage):
        self.folder = Path(location) / "recording"
        self.call("begin", Path(location), stage)

    def finish(self, stage, benchmark_completed):
        return self.call("finish", stage, benchmark_completed)

    def close(self):
        try:
            if self.thread.is_alive() and not self.failure:
                self.call("close")
            self.thread.join(timeout=10)
        finally:
            if self.cancelled.is_set() and self.folder:
                # Covers cancellation racing the worker's final metadata write.
                path = self.folder / "recording.json"
                with _METADATA_LOCK:
                    if path.exists():
                        record = json.loads(path.read_text(encoding="utf-8"))
                        record.update(state="interrupted", complete=False)
                        write_json(path, record)

    def worker(self):
        context, page, browser, manager, location = None, None, None, None, None
        pending = None
        try:
            from playwright.sync_api import sync_playwright

            manager = sync_playwright().start()
            browser = manager.chromium.launch(
                headless=not self.headed,
                args=[
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                ],
            )
            self.ready.set_result(True)
            while True:
                try:
                    name, args, pending = self.commands.get(timeout=0.05)
                except queue.Empty:
                    if page:
                        page.wait_for_timeout(50)
                    continue
                if name == "close":
                    pending.set_result(None)
                    break
                if name == "begin":
                    location, stage = args
                    folder = location / "recording"
                    folder.mkdir()
                    recording_start = time.monotonic()
                    save_recording(
                        folder / "recording.json",
                        {"state": "recording", "complete": False},
                    )
                    context = browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        device_scale_factor=1,
                        record_video_dir=str(folder),
                        record_video_size={"width": 1920, "height": 1080},
                    )
                    page = context.new_page()
                    errors = []
                    page.on(
                        "pageerror", lambda error, log=errors: log.append(str(error))
                    )
                    page.goto(self.url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_function(
                        "stage => window.monitorReady && window.currentStage === stage",
                        arg=stage,
                        timeout=30000,
                    )
                    pending.set_result(True)
                elif name == "finish":
                    stage, completed = args
                    page.wait_for_function(
                        "stage => window.currentStage === stage && window.finalPainted",
                        arg=stage,
                        timeout=15000,
                    )
                    page.wait_for_timeout(2000)
                    diagnostics = page.evaluate("window.monitorDiagnostics")
                    page.screenshot(path=str(folder / "final.png"))
                    video = page.video
                    ended = time.monotonic()
                    # Finalize the whole clip, including the statistics tail.
                    context.close()
                    video.save_as(str(folder / "viewport.webm"))
                    original = video.path()
                    if Path(original) != folder / "viewport.webm":
                        Path(original).unlink(missing_ok=True)
                    context, page = None, None
                    transcode(folder / "viewport.webm", folder / "viewport.mp4")
                    measured = inspect_video(folder / "viewport.mp4")
                    elapsed = ended - recording_start
                    completed = completed and not self.cancelled.is_set()
                    complete = recording_is_complete(
                        measured,
                        elapsed,
                        diagnostics,
                        errors,
                        benchmark_completed=completed,
                    )
                    result = {
                        "state": "interrupted"
                        if not completed
                        else "complete"
                        if complete
                        else "incomplete",
                        "complete": complete,
                        "benchmark_completed": completed,
                        "kind": "live_browser_viewport",
                        "wall_seconds": elapsed,
                        "video": "recording/viewport.mp4",
                        "raw_video": "recording/viewport.webm",
                        "final_image": "recording/final.png",
                        "video_probe": measured,
                        "browser": diagnostics,
                        "page_errors": errors,
                        "scope": "Live browser capture; preview FPS != VLM throughput.",
                    }
                    save_recording(folder / "recording.json", result)
                    pending.set_result(result)
                pending = None
        except BaseException as exc:
            self.failure = exc
            if location:
                save_recording(
                    location / "recording/recording.json",
                    {
                        "state": "failed",
                        "complete": False,
                        "error_type": type(exc).__name__,
                    },
                )
            if not self.ready.done():
                self.ready.set_exception(exc)
            if pending and not pending.done():
                pending.set_exception(exc)
        finally:
            # Keep any partial WebM on interruption; never label it complete.
            if context:
                context.close()
            if browser:
                browser.close()
            if manager:
                manager.stop()
