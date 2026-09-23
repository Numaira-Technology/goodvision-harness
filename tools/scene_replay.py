"""Replay observations through the packaged GoodVision event runtime."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goodvision_harness.evaluation.replay import (  # noqa: E402,F401
    MAX_LINE_BYTES,
    ReplayError,
    main,
    replay,
)

if __name__ == "__main__":
    raise SystemExit(main())
