"""Measure optional CV primitives on deterministic synthetic tracks, without models."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goodvision_harness.contracts import NormalizedBBox  # noqa: E402
from goodvision_harness.observations.normalization import from_snapshot  # noqa: E402
from goodvision_harness.perception.components import SupervisionPipeline  # noqa: E402
from goodvision_harness.perception.settings import (  # noqa: E402
    SupervisionSettings,
    Zone,
)
from goodvision_harness.rich_labeling import (  # noqa: E402
    RichLabelObject,
    RichLabelSnapshot,
)


def benchmark(*, frames=100, counts=(8, 64, 128)):
    import numpy as np
    import supervision as sv

    cases = []
    for count in counts:
        processor = SupervisionPipeline(
            source_id="synthetic",
            stream_session_id="benchmark",
            settings=SupervisionSettings(
                enabled=True,
                zones=(Zone("left", ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))),),
                lines=(Zone("middle", ((0.0, 0.5), (1.0, 0.5))),),
            ),
        )
        objects = tuple(
            RichLabelObject(
                str(i),
                "person",
                NormalizedBBox(
                    (i % 16) / 16,
                    (i // 16) / 8,
                    (i % 16 + 0.8) / 16,
                    (i // 16 + 0.8) / 8,
                ),
                0.9,
            )
            for i in range(count)
        )
        samples = []
        for sequence in range(frames + 10):
            observation = from_snapshot(
                RichLabelSnapshot("benchmark", sequence, sequence / 10, objects),
                source_id="synthetic",
                model="synthetic",
            )
            before = perf_counter()
            processor.process(observation, preserve_tracks=True)
            elapsed = (perf_counter() - before) * 1000
            if sequence >= 10:
                samples.append(elapsed)
        cases.append(
            {
                "objects": count,
                "frames": frames,
                "p50_ms": float(np.percentile(samples, 50)),
                "p95_ms": float(np.percentile(samples, 95)),
            }
        )
    return {
        "scope": (
            "synthetic CPU: conversion, polygon, line, smoothing; "
            "no inference/rendering"
        ),
        "supervision": sv.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 10000:
        parser.error("frames must be within 1..10000")
    print(json.dumps(benchmark(frames=args.frames), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
