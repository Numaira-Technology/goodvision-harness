"""Check local face analysis readiness."""

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "/opt/goodvision/runtime_gateway.py", "face"])


if __name__ == "__main__":
    raise SystemExit(main())
