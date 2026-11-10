#!/usr/bin/env python3
"""Download listed Drive videos, or check local inputs without model requests."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

if __package__:
    from .run_eval import frozen_inputs
else:
    from run_eval import frozen_inputs


def download_video(sample, output, links, token, drive_remote=None):
    filename = sample["filename"]
    if Path(filename).name != filename:
        raise ValueError("filename must not contain directories")
    destination = output / filename
    size = sample.get("size_bytes")
    if (destination.is_file() and destination.stat().st_size > 0
            and (size is None or destination.stat().st_size == size)):
        return

    file_id = sample.get("drive_file_id")
    if not file_id:
        reference = urlparse(sample.get("video_url") or sample.get("url", ""))
        parts = reference.path.strip("/").split("/")
        if reference.hostname == "drive.google.com" and parts[:2] == ["file", "d"] and len(parts) > 2:
            file_id = parts[2]
    url = links.get(filename) or links.get(file_id)
    headers = {}
    partial = destination.with_suffix(destination.suffix + ".part")
    if url:
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("download links must be HTTPS URLs")
    elif file_id and drive_remote:
        subprocess.run([
            "rclone", "backend", "copyid", drive_remote.rstrip(":") + ":",
            file_id, str(partial), "--ignore-times", "--ignore-checksum",
        ], check=True, capture_output=True)
        if partial.stat().st_size == 0:
            raise ValueError("download returned an empty file")
        if size is not None and partial.stat().st_size != size:
            raise ValueError("download size differs from the listed video")
        partial.replace(destination)
        return
    elif file_id and token:
        url = ("https://www.googleapis.com/drive/v3/files/"
               + file_id + "?alt=media&supportsAllDrives=true")
        headers["Authorization"] = "Bearer " + token
    else:
        raise ValueError(
            "provide a Drive file ID/URL and --drive-remote or "
            "GOOGLE_DRIVE_ACCESS_TOKEN, or a temporary download link")

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.headers.get_content_type() in ("text/html", "application/json"):
            raise ValueError("download returned a page or error instead of video")
        length = response.headers.get("Content-Length")
        with partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    if length is not None and partial.stat().st_size != int(length):
        raise ValueError("incomplete HTTP download")
    if partial.stat().st_size == 0:
        raise ValueError("download returned an empty file")
    if size is not None and partial.stat().st_size != size:
        raise ValueError("download size differs from the listed video")
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--samples", type=Path, help="question list with a samples array")
    source.add_argument("--videos", type=Path, help="media-only list with a videos array")
    parser.add_argument("--manifest", type=Path,
                        help="reviewed scoring manifest, required with --samples")
    parser.add_argument("--output", type=Path, required=True,
                        help="video directory on this machine")
    parser.add_argument("--links", type=Path,
                        help="private JSON map of filenames or Drive IDs "
                             "to temporary HTTPS URLs")
    parser.add_argument("--drive-remote",
                        help="existing rclone Drive remote name; no login is configured")
    parser.add_argument("--check-only", action="store_true",
                        help="check existing local videos with the runner; "
                             "no downloads or inference")
    args = parser.parse_args()
    if args.videos is None:
        missing = [flag for flag, value in (("--samples", args.samples), ("--manifest", args.manifest))
                   if value is None]
        if missing:
            parser.error("the following arguments are required: " + ", ".join(missing))
    elif args.check_only:
        parser.error("--check-only requires --samples and --manifest")
    if args.check_only:
        samples, media, _ = frozen_inputs(args.samples, args.manifest, [args.output])
        events = {}
        for sample in samples:
            events[sample["event"]] = events.get(sample["event"], 0) + 1
        print(json.dumps({"questions": len(samples), "videos": len(media),
                          "categories": events, "model_requests": 0}))
        return 0

    path, key = (args.videos, "videos") if args.videos else (args.samples, "samples")
    samples = json.loads(path.read_text(encoding="utf-8"))[key]
    links = json.loads(args.links.read_text(encoding="utf-8")) if args.links else {}
    token = os.environ.get("GOOGLE_DRIVE_ACCESS_TOKEN", "")
    if not links and not token and not args.drive_remote:
        parser.error("private Drive videos need --drive-remote, --links or GOOGLE_DRIVE_ACCESS_TOKEN")
    args.output.mkdir(parents=True, exist_ok=True)
    videos = {sample["filename"]: sample for sample in samples}
    failures = 0
    for filename, sample in videos.items():
        try:
            download_video(sample, args.output, links, token, args.drive_remote)
            print(filename + ": ready")
        except urllib.error.HTTPError as error:
            failures += 1
            print(f"{filename}: HTTP {error.code}", file=sys.stderr)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            failures += 1
            reason = (str(error) if isinstance(error, ValueError)
                      else type(error).__name__)
            print(f"{filename}: {reason}", file=sys.stderr)
    print(f"{len(videos) - failures}/{len(videos)} videos ready; {failures} failed")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
