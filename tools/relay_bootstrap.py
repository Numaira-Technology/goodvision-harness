"""Create the private cloud relay key without displaying or renting anything."""

from __future__ import annotations

import os
import secrets
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path


def valid_relay_key(value: str) -> bool:
    return (
        32 <= len(value) <= 512
        and value.isascii()
        and all(33 <= ord(character) <= 126 for character in value)
    )


def _stored_key(path: Path) -> str:
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(
                "Stored relay authentication key is unavailable or invalid"
            )
        with path.open("r", encoding="utf-8") as source:
            value = source.read(514)
    except (OSError, UnicodeError):
        raise ValueError(
            "Stored relay authentication key is unavailable or invalid"
        ) from None
    if len(value) > 513 or not valid_relay_key(value.strip()):
        raise ValueError("Stored relay authentication key is unavailable or invalid")
    return value.strip()


def _private_file(path: Path, root: bool) -> None:
    if root:
        os.chown(path, 0, 10001)
    path.chmod(0o640 if root else 0o600)


def controller_environment(
    environment: Mapping[str, str],
    directory: Path,
) -> dict[str, str]:
    result = dict(environment)
    if environment.get("SAM_ANALYSIS_ENABLED", "false").lower() != "true":
        return result
    key = environment.get("HARNESS_RELAY_KEY", "").strip()
    if key:
        if not valid_relay_key(key):
            raise ValueError(
                "HARNESS_RELAY_KEY must contain 32-512 ASCII nonspace characters"
            )
    else:
        filename = environment.get("HARNESS_RELAY_KEY_FILE", "").strip()
        path = Path(filename) if filename else directory / "relay-key"
        root = getattr(os, "geteuid", lambda: -1)() == 0
        created_parent = not path.parent.exists()
        owned_parent = path == Path("/relay-auth/key")
        if not path.exists():
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
                mode=0o750 if root else 0o700,
            )
            if created_parent or owned_parent:
                if root:
                    # Only the dedicated managed directory gets gateway access;
                    # never change an existing operator-selected parent.
                    os.chown(path.parent, 0, 10001)
                path.parent.chmod(0o750 if root else 0o700)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".relay-key-", dir=path.parent
            )
            staging = Path(temporary)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                    target.write(secrets.token_hex(32))
                    target.flush()
                    os.fsync(target.fileno())
                _private_file(staging, root)
                try:
                    # Publish the complete key atomically, without replacing a
                    # key another controller has already created.
                    os.link(staging, path)
                except FileExistsError:
                    pass
            finally:
                staging.unlink(missing_ok=True)
        key = _stored_key(path)
    result["HARNESS_RELAY_KEY"] = key
    return result
