"""Entry point for fixed-evidence, twelve-stream event quality comparisons."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vlm_benchmark.behavior import main

if __name__ == "__main__":
    main()
