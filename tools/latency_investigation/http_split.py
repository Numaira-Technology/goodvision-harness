"""Optional upload / first-body-byte / read-done timings for urllib clients.

Set ``GOODVISION_SPLIT_TIMING=1`` to write ``transport-splits.jsonl``. When the
variable is unset the caller keeps using ``urllib.request.urlopen`` and writes
nothing. The formal receipt, ledger elapsed time, and transport_seconds are
not given these fields.

The three marks follow the same phases as ``scene_model``'s HTTP trace:
body send complete, first response-body read, and body read finished. This
client does not stream, so a body that arrives in one ``read`` records the
same timestamp for first byte and read done.
"""

from __future__ import annotations

import contextvars
import http.client
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_UNIT = contextvars.ContextVar("gv_split_request", default=None)
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
SIDECAR = "transport-splits.jsonl"


def split_timing_enabled() -> bool:
    return os.environ.get("GOODVISION_SPLIT_TIMING", "").strip() == "1"


def bind_request(unit_id, attempt=None):
    return _UNIT.set({"unit": unit_id, "attempt": attempt})


def reset_request(token) -> None:
    _UNIT.reset(token)


class TimingHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, timings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._timings = timings if isinstance(timings, dict) else {}
        self._t0 = time.monotonic()

    def endheaders(self, message_body=None, *, encode_chunked=False):
        super().endheaders(message_body, encode_chunked=encode_chunked)
        self._timings["upload_done_seconds"] = time.monotonic() - self._t0

    def getresponse(self):
        response = super().getresponse()
        if getattr(response, "_gv_split_wrapped", False):
            return response
        original = response.read

        def read(amt=None):
            data = original() if amt is None else original(amt)
            now = time.monotonic() - self._t0
            if data and "first_byte_seconds" not in self._timings:
                self._timings["first_byte_seconds"] = now
            # read() with no size, or a short read, is the end of the body.
            if amt is None or data == b"" or (isinstance(amt, int) and len(data) < amt):
                self._timings["read_done_seconds"] = now
            return data

        response.read = read
        response._gv_split_wrapped = True
        return response


class TimingHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, timings):
        super().__init__()
        self._timings = timings

    def http_open(self, req):
        timings = self._timings

        def factory(host, port=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, **kwargs):
            return TimingHTTPConnection(
                host, port=port, timeout=timeout, timings=timings, **kwargs
            )

        return self.do_open(factory, req)


def begin_open(request, timeout):
    """Open ``request``. Returns ``(response, timings)``.

    ``timings`` is None unless split timing is enabled. ``HTTPError`` is raised
    the same way ``urlopen`` raises it, with the partial timings on
    ``_gv_timings``.
    """
    if not split_timing_enabled():
        return urllib.request.urlopen(request, timeout=timeout), None
    timings: dict = {}
    opener = urllib.request.build_opener(TimingHTTPHandler(timings))
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        error._gv_timings = timings
        raise
    return response, timings


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def record_split(directory, timings, *, http_status=None) -> None:
    if not isinstance(timings, dict):
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bound = _UNIT.get() or {}
    row = {
        "unit": bound.get("unit"),
        "attempt": bound.get("attempt"),
        "http_status": http_status,
        "upload_done_seconds": timings.get("upload_done_seconds"),
        "first_byte_seconds": timings.get("first_byte_seconds"),
        "read_done_seconds": timings.get("read_done_seconds"),
    }
    path = directory / SIDECAR
    line = json.dumps(row, sort_keys=True) + "\n"
    with _lock_for(path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
