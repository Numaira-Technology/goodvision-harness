"""Call a remote Contour node with JPEG frames. Uses only Python's standard library."""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Send JPEG frames to a Contour node")
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("frames", type=Path, nargs="+")
    args = parser.parse_args()
    if not 0.5 <= args.fps <= 30:
        parser.error("fps must be between 0.5 and 30")
    key = (
        args.key_file.read_text().strip()
        if args.key_file
        else os.environ.get("CONTOUR_API_KEY", "")
    )
    if not key:
        parser.error("set CONTOUR_API_KEY or use --key-file")

    # Do not forward a key through an unexpected redirect.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)

    def request(method, path, value=None):
        data = None if value is None else json.dumps(value).encode()
        req = urllib.request.Request(
            args.url.rstrip("/") + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
        )
        with opener.open(req, timeout=80) as response:
            return json.load(response)

    request("GET", "/health/ready")
    stream_id = "camera-" + uuid.uuid4().hex
    opened = request("POST", "/v1/contour/sessions", {"stream_session_id": stream_id})
    session = "/v1/contour/sessions/" + opened["video_session_id"]
    try:
        for index, path in enumerate(args.frames):
            if path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("JPEG exceeds the 8 MiB transport limit")
            body = {
                "stream_session_id": stream_id,
                "request_id": uuid.uuid4().hex,
                "is_last_batch": index == len(args.frames) - 1,
                "frames": [
                    {
                        "frame_sequence": index,
                        "timestamp_seconds": index / args.fps,
                        "jpeg_base64": base64.b64encode(path.read_bytes()).decode(),
                    }
                ],
            }
            result = request("POST", session + "/append", body)
            print(json.dumps(result, ensure_ascii=False))
    finally:
        # No automatic retry of uncertain inference. Best-effort session cleanup.
        try:
            request("DELETE", session)
        except OSError:
            pass


if __name__ == "__main__":
    main()
