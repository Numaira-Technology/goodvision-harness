"""Create a local node key without printing it or replacing an existing key."""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def initialize(root: Path):
    directory = root / "secrets"
    # On Windows inherit the workspace ACL so the signed-in Docker Desktop user
    # can mount the secret even when a helper account initializes this folder.
    directory.mkdir(
        mode=0o777 if os.name == "nt" else 0o700, parents=True, exist_ok=True
    )
    path = directory / "node-key"
    try:
        # Container compose secrets are bind mounts on some platforms. The file
        # must be readable by the non-root node; host directory stays owner-only.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(36) + "\n")
    (root / "models").mkdir(exist_ok=True)
    return path


if __name__ == "__main__":
    result = initialize(Path(__file__).resolve().parent)
    print(f"Node initialized. Access key saved at: {result}")
