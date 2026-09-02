"""Check object detection and appearance readiness."""

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "/opt/goodvision/runtime_gateway.py", "perception"])


if __name__ == "__main__":
    raise SystemExit(main())
