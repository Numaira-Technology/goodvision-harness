"""Check the installed application runtime."""

import subprocess
import sys


def main() -> int:
    return subprocess.call([sys.executable, "/opt/goodvision/runtime_gateway.py", "runtime"])


if __name__ == "__main__":
    raise SystemExit(main())
