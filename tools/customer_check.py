"""Launch the installed customer validation entrypoint without a parent shim."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ENTRYPOINTS = {
    "check_runtime.py": "runtime",
    "check_face.py": "face",
    "check_perception.py": "perception",
}


def _failure(check: str, message: str, returncode: int) -> int:
    print(json.dumps({
        "check": check, "status": "failed", "message": message,
    }, separators=(",", ":")), flush=True)
    return returncode


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    check = _ENTRYPOINTS.get(Path(__file__).name)
    if args or check is None:
        return _failure("request", "invalid validation entrypoint", 2)
    try:
        # Keep the original process identity so stop signals reach the gateway.
        os.execv(sys.executable, [
            sys.executable, "/opt/goodvision/runtime_gateway.py", check,
        ])
    except FileNotFoundError:
        return _failure(check, "validation launcher is unavailable", 127)
    except OSError:
        return _failure(check, "validation launcher could not start", 126)
    except KeyboardInterrupt:
        return _failure(check, "validation was interrupted", 130)
    # A successful exec never returns; fail closed if an abnormal runtime does.
    return _failure(check, "validation launcher did not replace its process", 1)


if __name__ == "__main__":
    raise SystemExit(main())
