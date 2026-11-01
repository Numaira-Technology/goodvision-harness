#!/usr/bin/env python3
"""Occupy N CPU cores with busy work to simulate full-annotation CPU pressure.

Full annotation (overlay re-encode at 30fps, extra JPEG encode for the SAM
upload, etc.) saturates cores and starves the aiortc receive loop. This is a
crude stand-in so we can reproduce that contention without a GPU.

    python diag/cpu_load.py --cores 8 --seconds 60
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time


def _spin(deadline: float) -> None:
    x = 0.0
    while time.monotonic() < deadline:
        # tight FP loop; no GIL relief -> real core occupancy
        for _ in range(200000):
            x = x * 1.000001 + 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()
    deadline = time.monotonic() + args.seconds
    procs = [mp.Process(target=_spin, args=(deadline,)) for _ in range(args.cores)]
    print(f"[cpu_load] pinning {args.cores} cores for {args.seconds:.0f}s")
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("[cpu_load] done")


if __name__ == "__main__":
    main()
