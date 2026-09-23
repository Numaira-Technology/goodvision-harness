"""Synthetic runtime cost audit; no model, camera, GPU, network or media I/O.

Each case runs in a fresh interpreter. RSS is a high-water mark, not allocated
bytes or an estimate of total deployment RAM. Optional --baseline-source points
to an extracted older src/ tree and uses exactly the same installed dependencies.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def peak_mib():
    import resource  # macOS/Linux only; this audit does not estimate Windows RSS.

    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def settings(profile):
    from goodvision_harness.perception.settings import SupervisionSettings, Zone

    count = {"empty": 0, "typical": 1, "maximum": 16}[profile]
    return SupervisionSettings(
        enabled=True,
        zones=tuple(
            Zone(f"zone-{i}", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
            for i in range(count)
        ),
        lines=tuple(Zone(f"line-{i}", ((0.0, 0.5), (1.0, 0.5))) for i in range(count)),
    )


def worker(case):
    sys.path.insert(0, case["source"])
    started = time.perf_counter()
    import goodvision_harness.service  # noqa: F401
    from goodvision_harness.contracts import NormalizedBBox
    from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
    from goodvision_harness.scene_runtime import SceneRuntime

    service_import_ms = (time.perf_counter() - started) * 1000
    before_mib = peak_mib()
    count = case.get("objects", 64)
    objects = tuple(
        RichLabelObject(
            str(i),
            "person",
            NormalizedBBox(
                (i % 16) / 16, (i // 16) / 8, (i % 16 + 0.8) / 16, (i // 16 + 0.8) / 8
            ),
            0.9,
        )
        for i in range(count)
    )
    options = (
        {}
        if case["profile"] == "off"
        else {
            "supervision_settings": settings(case["profile"]),
        }
    )
    started = time.perf_counter()
    scenes = [
        SceneRuntime(f"session-{i}", f"camera-{i}", event_labels=(), **options)
        for i in range(case.get("streams", 1))
    ]
    initialization_ms = (time.perf_counter() - started) * 1000

    def snapshot(sequence, index=0):
        return RichLabelSnapshot(f"session-{index}", sequence, sequence / 7.5, objects)

    kind = case["kind"]
    result = {}
    if kind == "lock":
        entered, release, reader_done = (threading.Event() for _ in range(3))
        scene = scenes[0]
        original = scene._geometry_pipeline.process

        def stall(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("audit did not release injected stall")
            return original(*args, **kwargs)

        scene._geometry_pipeline.process = stall
        writer = threading.Thread(target=lambda: scene.observe_tracks(snapshot(0)))
        elapsed = []

        def read_scene():
            before = time.perf_counter()
            scene.snapshot(include_history=False)
            elapsed.append((time.perf_counter() - before) * 1000)
            reader_done.set()

        reader = threading.Thread(target=read_scene)
        writer.start()
        try:
            if not entered.wait(2):
                raise RuntimeError("geometry was never entered")
            reader.start()
            result["snapshot_completed_during_200ms_stall"] = reader_done.wait(0.2)
        finally:
            release.set()
            writer.join(3)
            if reader.ident is not None:
                reader.join(3)
        if writer.is_alive() or reader.is_alive() or not elapsed:
            raise RuntimeError("audit thread failed to finish")
        result["snapshot_wait_ms"] = elapsed[0]
    else:
        call = scenes[0].observe_tracks
        if kind == "primitive":
            from goodvision_harness.observations.normalization import from_snapshot

            pipeline = scenes[0]._geometry_pipeline

            def call(value):
                return pipeline.process(
                    from_snapshot(value, source_id="camera-0", model="synthetic"),
                    preserve_tracks=True,
                )

        wall, cpu = [], []
        frames = 32 if kind == "memory" else case["frames"] + 20
        for sequence in range(frames):
            if kind == "memory":
                for i, scene in enumerate(scenes):
                    scene.observe_tracks(snapshot(sequence, i))
            else:
                value = snapshot(sequence)
                before_cpu, before_wall = time.process_time(), time.perf_counter()
                call(value)
                elapsed_wall = (time.perf_counter() - before_wall) * 1000
                elapsed_cpu = (time.process_time() - before_cpu) * 1000
                if sequence >= 20:
                    wall.append(elapsed_wall)
                    cpu.append(elapsed_cpu)
        if wall:
            import numpy as np

            result.update(
                p50_ms=float(np.percentile(wall, 50)),
                p95_ms=float(np.percentile(wall, 95)),
                mean_cpu_ms=statistics.mean(cpu),
            )
    for scene in scenes:
        failures = getattr(scene, "_geometry_failures", 0)
        if failures:
            raise RuntimeError(f"geometry failed {failures} times")
    result.update(
        peak_rss_mib=peak_mib(),
        service_only_peak_rss_mib=before_mib,
        service_import_ms=service_import_ms,
        initialization_ms=initialization_ms,
        supervision_imported="supervision" in sys.modules,
        gpu_frameworks_imported=[
            m for m in ("torch", "tensorflow", "cupy") if m in sys.modules
        ],
    )
    return result


def audit(args):
    cases = []
    source = str(ROOT / "src")
    for profile in ("off", "empty", "typical", "maximum"):
        for streams in (1, 12):
            for repeat in range(3):
                cases.append(
                    dict(
                        kind="memory",
                        profile=profile,
                        streams=streams,
                        objects=64,
                        repeat=repeat,
                        source=source,
                    )
                )
    for count in (8, 64, 128):
        for profile in ("off", "empty", "typical", "maximum"):
            cases.append(
                dict(
                    kind="scene",
                    profile=profile,
                    objects=count,
                    frames=args.frames,
                    source=source,
                )
            )
        for profile in ("empty", "typical", "maximum"):
            cases.append(
                dict(
                    kind="primitive",
                    profile=profile,
                    objects=count,
                    frames=args.frames,
                    source=source,
                )
            )
        if args.baseline_source:
            cases.append(
                dict(
                    kind="scene",
                    profile="off",
                    objects=count,
                    frames=args.frames,
                    source=str(args.baseline_source),
                )
            )
    cases.append(dict(kind="lock", profile="typical", objects=8, source=source))
    outputs = []
    for case in cases:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                json.dumps(case),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )
        # Do not retain absolute local filesystem paths in the shareable result.
        public_case = {k: v for k, v in case.items() if k != "source"}
        public_case["tree"] = "experiment" if case["source"] == source else "baseline"
        outputs.append({**public_case, **json.loads(completed.stdout)})
    return dict(
        scope=__doc__,
        python=platform.python_version(),
        platform=platform.platform(),
        cases=outputs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform not in {"darwin", "linux"}:
        parser.error("RSS measurement currently requires macOS or Linux")
    if not 20 <= args.frames <= 2000:
        parser.error("frames must be within 20..2000")
    if args.baseline_source and not args.baseline_source.is_dir():
        parser.error("baseline source must be an existing src/ directory")
    print(
        json.dumps(
            worker(json.loads(args.worker)) if args.worker else audit(args),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
