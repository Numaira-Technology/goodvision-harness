"""Fetch only the pinned Git source commits needed by offline native regressions.

This explicit CI preparation step downloads repository source, never model weights
or environments. Export and test code remain offline and verify every source hash.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path(__file__).with_name("native_assets") / "sources.json"


def fetch_sources(repository=ROOT, remote="origin"):
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    commits = [spec["commit"] for spec in sources.values()]
    if not commits or any(not isinstance(commit, str) or
                          not re.fullmatch(r"[0-9a-f]{40}", commit) for commit in commits):
        raise ValueError("native source pins must be full commit SHA-1 values")
    commits = sorted(set(commits))
    subprocess.run(["git", "-C", str(repository), "fetch", "--no-tags", "--depth=1",
                    "--", remote, *commits], check=True)
    return commits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args(argv)
    fetch_sources(args.repository, args.remote)


if __name__ == "__main__":
    main()
