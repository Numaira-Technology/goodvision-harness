"""Explicitly prepare the pinned native VideoJev cache; never launch a model."""

import argparse
import json
from pathlib import Path

from .contract import MODEL, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download or reuse the model snapshot")
    download.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download

        path = snapshot_download(MODEL, revision=REVISION, cache_dir=str(args.cache_dir))
        print(json.dumps({"model": MODEL, "revision": REVISION, "snapshot": path}), flush=True)


if __name__ == "__main__":
    main()
