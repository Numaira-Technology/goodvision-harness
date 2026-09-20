"""Reject tracked files excluded by the repository's own .gitignore rules."""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    # Use only versioned per-directory rules: a developer's global excludes or
    # .git/info/exclude must not change what is accepted by CI. --cached also
    # catches ignored files deliberately added with `git add --force`.
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--ignored",
            "--exclude-per-directory=.gitignore",
            "-z",
        ],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = result.stdout.decode("utf-8", errors="replace").split("\0")
    ignored = sorted(path for path in paths if path)
    if ignored:
        print("Tracked files match repository ignore rules:")
        for path in ignored:
            print(f"  {path}")
        print(
            "Remove generated/private files from the index with git rm --cached, "
            "or add a narrow .gitignore exception for a reviewed source asset."
        )
        return 1
    print("Repository hygiene: no ignored files are tracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
