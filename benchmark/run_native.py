"""Checkout entrypoint: python benchmark/run_native.py --help."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench.native import main

if __name__ == "__main__":
    raise SystemExit(main())
