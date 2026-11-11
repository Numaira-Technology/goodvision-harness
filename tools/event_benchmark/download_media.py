#!/usr/bin/env python3
"""Download listed Drive videos, or check local inputs without model requests."""

import argparse
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if __package__:
    from .run_eval import frozen_inputs
else:
    from run_eval import frozen_inputs


def video_metadata(sample):
    """Validate transfer metadata without modifying frozen source records."""
    if not isinstance(sample, dict):
        raise ValueError("each video must be an object")
    filename = sample.get("filename")
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or any(c in filename for c in "/\\:")
        or any(ord(c) < 32 for c in filename)
    ):
        raise ValueError("filename must be a plain local filename")
    size = sample.get("size_bytes")
    if size is not None and (type(size) is not int or size <= 0):
        raise ValueError("size_bytes must be a positive integer when supplied")
    file_id = sample.get("drive_file_id")
    if not file_id:
        reference = sample.get("video_url") or sample.get("url")
        if reference:
            if not isinstance(reference, str):
                raise ValueError("Drive reference must be a string")
            if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", reference):
                file_id = reference
            else:
                parsed = urlparse(reference)
                if (
                    parsed.scheme == "https"
                    and parsed.hostname == "drive.google.com"
                    and parsed.port in {None, 443}
                    and parsed.username is None
                ):
                    parts = parsed.path.strip("/").split("/")
                    if (
                        parts[:2] == ["file", "d"]
                        and len(parts) in {3, 4}
                        and (len(parts) == 3 or parts[3] in {"view", "preview", "edit"})
                    ):
                        file_id = parts[2]
                    elif parsed.path in {"/open", "/uc", "/download"}:
                        ids = parse_qs(parsed.query).get("id", [])
                        if len(ids) == 1:
                            file_id = ids[0]
    if file_id is not None and (
        not isinstance(file_id, str)
        or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", file_id)
    ):
        raise ValueError("invalid Drive file ID")
    return {"filename": filename, "size_bytes": size, "drive_file_id": file_id}


def unique_videos(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("the video list must be a nonempty array")
    videos = {}
    for row in rows:
        sample = video_metadata(row)
        filename = sample["filename"]
        previous = videos.setdefault(filename, sample)
        for key in ("size_bytes", "drive_file_id"):
            if previous[key] is not None and sample[key] is not None:
                if previous[key] != sample[key]:
                    raise ValueError("conflicting download metadata for " + filename)
            elif sample[key] is not None:
                previous[key] = sample[key]
    return videos


def drive_remote_name(remote):
    if not isinstance(remote, str) or not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_ .-]*:?", remote
    ):
        raise ValueError("--drive-remote must name an existing rclone remote")
    return remote.removesuffix(":") + ":"


def require_https(url):
    if not isinstance(url, str):
        raise ValueError("download links must be HTTPS URLs")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None:
        raise ValueError("download links must be HTTPS URLs without user information")
    return parsed


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = require_https(newurl)
        original = urlparse(req.full_url)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and (original.hostname, original.port or 443) != (
            target.hostname,
            target.port or 443,
        ):
            redirected.remove_header("Authorization")
        return redirected


def open_download(request):
    return urllib.request.build_opener(HTTPSRedirectHandler()).open(request, timeout=90)


def download_video(sample, output, links, token, drive_remote=None):
    sample = video_metadata(sample)
    filename, size = sample["filename"], sample["size_bytes"]
    destination = output / filename
    if destination.is_symlink():
        raise ValueError("destination must not be a symbolic link")
    if destination.is_file():
        current_size = destination.stat().st_size
        if current_size > 0 and (size is None or current_size == size):
            return

    file_id = sample["drive_file_id"]
    has_link = filename in links or file_id in links
    url = links[filename] if filename in links else links.get(file_id)
    headers = {}
    remote = None
    if has_link:
        require_https(url)
    elif file_id and drive_remote:
        remote = drive_remote_name(drive_remote)
    elif file_id and token:
        url = (
            "https://www.googleapis.com/drive/v3/files/"
            + file_id
            + "?alt=media&supportsAllDrives=true"
        )
        headers["Authorization"] = "Bearer " + token
    else:
        raise ValueError(
            "provide a Drive file ID/URL and --drive-remote or "
            "GOOGLE_DRIVE_ACCESS_TOKEN, or a temporary download link"
        )

    # A fresh private staging directory cannot reuse a stale .part file or follow
    # a pre-existing .part symlink. Publish only a complete, validated transfer.
    with tempfile.TemporaryDirectory(prefix=".event-download-", dir=output) as staging:
        partial = Path(staging) / "video.part"
        length = None
        if remote:
            subprocess.run(
                [
                    "rclone",
                    "backend",
                    "copyid",
                    remote,
                    file_id,
                    str(partial.resolve()),
                ],
                check=True,
                capture_output=True,
            )
        else:
            request = urllib.request.Request(url, headers=headers)
            with open_download(request) as response:
                if response.headers.get_content_type() in (
                    "text/html",
                    "application/json",
                ):
                    raise ValueError(
                        "download returned a page or error instead of video"
                    )
                length = response.headers.get("Content-Length")
                with partial.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
        transferred = partial.stat().st_size
        if length is not None and transferred != int(length):
            raise ValueError("incomplete HTTP download")
        if transferred == 0:
            raise ValueError("download returned an empty file")
        if size is not None and transferred != size:
            raise ValueError("download size differs from the listed video")
        partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--samples", type=Path, help="question list with a samples array"
    )
    source.add_argument(
        "--videos", type=Path, help="media-only list with a videos array"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="reviewed scoring manifest, required with --samples",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="video directory on this machine"
    )
    parser.add_argument(
        "--links",
        type=Path,
        help="private JSON map of filenames or Drive IDs to temporary HTTPS URLs",
    )
    parser.add_argument(
        "--drive-remote",
        help="existing rclone Drive remote name; no login is configured",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="check existing local videos with the runner; no downloads or inference",
    )
    args = parser.parse_args()
    if args.videos is None:
        missing = [
            flag
            for flag, value in (
                ("--samples", args.samples),
                ("--manifest", args.manifest),
            )
            if value is None
        ]
        if missing:
            parser.error("the following arguments are required: " + ", ".join(missing))
    elif args.check_only:
        parser.error("--check-only requires --samples and --manifest")
    if args.check_only:
        samples, media, _ = frozen_inputs(args.samples, args.manifest, [args.output])
        events = {}
        for sample in samples:
            events[sample["event"]] = events.get(sample["event"], 0) + 1
        print(
            json.dumps(
                {
                    "questions": len(samples),
                    "videos": len(media),
                    "categories": events,
                    "model_requests": 0,
                }
            )
        )
        return 0

    path, key = (args.videos, "videos") if args.videos else (args.samples, "samples")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))[key]
        videos = unique_videos(rows)
        links = json.loads(args.links.read_text(encoding="utf-8")) if args.links else {}
        if not isinstance(links, dict):
            raise ValueError("--links must contain a JSON object")
        if args.drive_remote is not None:
            drive_remote_name(args.drive_remote)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    token = os.environ.get("GOOGLE_DRIVE_ACCESS_TOKEN", "")
    if not links and not token and not args.drive_remote:
        parser.error(
            "private Drive videos need --drive-remote, --links or "
            "GOOGLE_DRIVE_ACCESS_TOKEN"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for filename, sample in videos.items():
        try:
            download_video(sample, args.output, links, token, args.drive_remote)
            print(filename + ": ready")
        except urllib.error.HTTPError as error:
            failures += 1
            print(f"{filename}: HTTP {error.code}", file=sys.stderr)
        except (
            OSError,
            ValueError,
            http.client.HTTPException,
            subprocess.SubprocessError,
        ) as error:
            failures += 1
            reason = (
                str(error) if isinstance(error, ValueError) else type(error).__name__
            )
            print(f"{filename}: {reason}", file=sys.stderr)
    print(f"{len(videos) - failures}/{len(videos)} videos ready; {failures} failed")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
