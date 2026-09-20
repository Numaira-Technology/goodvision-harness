"""Isolated FFmpeg receive/color conversion with a timestamped local frame pipe.

The subprocess owns the network decoder. Its NUT output retains source PTS and
contains BGR pixels, so PyAV does not repeat HDR conversion in the app process.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
from typing import Any


class DecodeSupportError(RuntimeError):
    pass


def color_filter(metadata: dict[str, Any], fps: float) -> str:
    # Select in source time before expensive color work. A timestamp reset
    # begins a new sampling phase instead of suppressing the restarted stream.
    slot = f"floor(t*{fps:.9g}+0.01)"
    previous = f"floor(prev_selected_t*{fps:.9g}+0.01)"
    sampling = (
        "select='isnan(t)+isnan(prev_selected_t)+lt(t,prev_selected_t)"
        f"+gt({slot},{previous})'"
    )
    if metadata.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        # zscale uses the actual source matrix, range, primaries and transfer.
        # Keep float values through linear-light tone mapping; tag only AFTER
        # transforming to BT.709. Never relabel HDR pixels as SDR.
        conversion = (
            "zscale=t=linear:npl=100,format=gbrpf32le,"
            "tonemap=tonemap=hable:desat=0,"
            "zscale=p=bt709:t=bt709:m=bt709:r=limited,"
            "format=yuv420p,format=bgr24"
        )
    else:
        conversion = "format=bgr24"
    return f"{sampling},{conversion}"


class FFmpegReceiver:
    """One connection owner, finite I/O waits, bounded diagnostics, no shell."""

    def __init__(self, url: str, protocol: str, fps: float):
        self.url, self.protocol, self.fps = url, protocol, fps
        self.metadata: dict[str, Any] = {}
        self.decode_errors = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None

    def _start(self, args: list[str]) -> subprocess.Popen:
        with self._lock:
            if self._stop.is_set():
                raise InterruptedError("video receiver stopped")
            process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
            )
            self._process = process
            return process

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            except ProcessLookupError:
                pass

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            process = self._process
        # Interrupt blocked native pipe reads by ending their producer. Closing
        # the PyAV container from this other thread would race its decoder.
        if process is not None:
            self._terminate(process)

    def _input_options(self) -> list[str]:
        options = ["-rw_timeout", "5000000"]
        if self.protocol in {"rtsp", "rtsps"}:
            options.extend(["-rtsp_transport", "tcp"])
        return options

    def _probe(self) -> None:
        process = self._start(
            [
                "ffprobe",
                "-v",
                "error",
                *self._input_options(),
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,pix_fmt,color_space,color_transfer,"
                "color_primaries,color_range,avg_frame_rate",
                "-of",
                "json",
                self.url,
            ]
        )
        try:
            output, _ = process.communicate(timeout=12)
            if process.returncode:
                raise RuntimeError("video source probe failed")
            streams = json.loads(output).get("streams", [])
            if not streams:
                raise RuntimeError("video source has no video stream")
            self.metadata = streams[0]
        except subprocess.TimeoutExpired:
            raise RuntimeError("video source probe timed out") from None
        finally:
            self._terminate(process)
            with self._lock:
                self._process = None

    def _drain_errors(self, process: subprocess.Popen) -> None:
        # Do not retain stderr: source URLs can contain camera credentials.
        # read(size) also bounds memory for malformed streams without newlines.
        while process.stderr.read(4096):
            self.decode_errors += 1

    def frames(self) -> Iterator[Any]:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise DecodeSupportError("FFmpeg receiver requires ffmpeg and ffprobe")
        try:
            import av
        except ImportError as exc:
            raise DecodeSupportError(
                "FFmpeg receiver requires the video extra"
            ) from exc
        self._probe()
        process = self._start(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                *self._input_options(),
                "-threads",
                "2",
                "-err_detect",
                "explode",
                "-fflags",
                "+discardcorrupt",
                "-i",
                self.url,
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-filter_threads",
                "2",
                "-vf",
                color_filter(self.metadata, self.fps),
                "-c:v",
                "rawvideo",
                "-threads:v",
                "1",
                "-pix_fmt",
                "bgr24",
                "-fps_mode",
                "passthrough",
                "-f",
                "nut",
                "-flush_packets",
                "1",
                "pipe:1",
            ]
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_errors,
            args=(process,),
            name="video-decoder-errors",
            daemon=True,
        )
        self._stderr_thread.start()
        try:
            with av.open(
                process.stdout,
                format="nut",
                options={
                    "probesize": "32768",
                    "analyzeduration": "0",
                },
            ) as container:
                for frame in container.decode(video=0):
                    if self._stop.is_set():
                        return
                    if frame.is_corrupt:
                        self.decode_errors += 1
                        continue
                    yield frame
            if not self._stop.is_set() and process.wait(timeout=2):
                raise RuntimeError("video receiver exited unsuccessfully")
        finally:
            self._terminate(process)
            self._stderr_thread.join(timeout=1)
            process.stdout.close()
            process.stderr.close()
            with self._lock:
                self._process = None
