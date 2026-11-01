"""Source identity for portable test bundles, including uncommitted extensions."""

import json
import subprocess
from pathlib import Path

INTEGRATION = "main-grounding-isolated-benchmarks-v3"
BASE = "9dbd5103a5cc1c4a7886d7c80492f70d8abe1e63"


def source_identity(root):
    root = Path(root)
    try:
        head = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {
            "integration": INTEGRATION,
            "base_main": BASE,
            "checkout_commit": head,
            "uncommitted_changes": dirty,
            "identity_scope": "commit plus per-file source hashes",
        }
    except (OSError, subprocess.CalledProcessError):
        path = root / "bundle-manifest.json"
        if path.is_file():
            return json.loads(path.read_text()).get(
                "source_identity",
                {
                    "integration": INTEGRATION,
                    "checkout_commit": None,
                    "identity_scope": "per-file source hashes only",
                },
            )
        return {
            "integration": INTEGRATION,
            "checkout_commit": None,
            "identity_scope": "per-file source hashes only",
        }
