#!/usr/bin/env python3
"""Download listed Drive videos, or check local inputs without model requests."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.error
import urllib.request

if __package__:
    from .run_eval import frozen_inputs
else:
    from run_eval import frozen_inputs


DATASETS = Path(__file__).resolve().parent / "datasets"


def download_video(sample, output, links, token):
    filename = sample["filename"]
    if Path(filename).name != filename:
        raise ValueError("filename must not contain directories")
    destination = output / filename
    size = sample["size_bytes"]
    if destination.is_file() and destination.stat().st_size == size:
        return

    url = links.get(filename) or links.get(sample["drive_file_id"])
    headers = {}
    if url:
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("download links must be HTTPS URLs")
    elif token:
        url = ("https://www.googleapis.com/drive/v3/files/"
               + sample["drive_file_id"] + "?alt=media&supportsAllDrives=true")
        headers["Authorization"] = "Bearer " + token
    else:
        raise ValueError("provide a temporary download link or GOOGLE_DRIVE_ACCESS_TOKEN")

    request = urllib.request.Request(url, headers=headers)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.headers.get_content_type() in ("text/html", "application/json"):
            raise ValueError("download returned a page or error instead of video")
        with partial.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    if partial.stat().st_size != size:
        raise ValueError("download size differs from the listed video")
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DATASETS / "samples_robot41.json")
    parser.add_argument("--manifest", type=Path, default=DATASETS / "manifest_robot41.json")
    parser.add_argument("--output", type=Path, required=True, help="video directory on this machine")
    parser.add_argument("--links", type=Path,
                        help="private JSON map of filenames or Drive IDs to temporary HTTPS URLs")
    parser.add_argument("--check-only", action="store_true",
                        help="check existing local videos with the runner; no downloads or inference")
    args = parser.parse_args()
    if args.check_only:
        samples, media, _ = frozen_inputs(args.samples, args.manifest, [args.output])
        events = {}
        for sample in samples:
            events[sample["event"]] = events.get(sample["event"], 0) + 1
        print(json.dumps({"questions": len(samples), "videos": len(media),
                          "categories": events, "model_requests": 0}))
        return 0

    samples = json.loads(args.samples.read_text(encoding="utf-8"))["samples"]
    links = json.loads(args.links.read_text(encoding="utf-8")) if args.links else {}
    token = os.environ.get("GOOGLE_DRIVE_ACCESS_TOKEN", "")
    if not links and not token:
        parser.error("private Drive videos need --links or GOOGLE_DRIVE_ACCESS_TOKEN")
    args.output.mkdir(parents=True, exist_ok=True)
    videos = {sample["filename"]: sample for sample in samples}
    failures = 0
    for filename, sample in videos.items():
        try:
            download_video(sample, args.output, links, token)
            print(filename + ": ready")
        except urllib.error.HTTPError as error:
            failures += 1
            print(f"{filename}: HTTP {error.code}", file=sys.stderr)
        except (OSError, ValueError) as error:
            failures += 1
            reason = str(error) if isinstance(error, ValueError) else type(error).__name__
            print(f"{filename}: {reason}", file=sys.stderr)
    print(f"{len(videos) - failures}/{len(videos)} videos ready; {failures} failed")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
