"""All commands default to planning; --execute is required for runtime work."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "plan",
            "bundle",
            "prepare",
            "start",
            "preflight",
            "run",
            "analyze",
            "collect",
        ],
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--volume", type=Path, default=Path("/models"))
    parser.add_argument("--video", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args(argv)
    if args.action == "plan" or not args.execute:
        from .benchmark import plan

        print(json.dumps(plan(), indent=2))
        return 0

    if args.action in {"bundle", "run", "collect"} and args.output is None:
        parser.error("--output is required")
    if args.action in {"preflight", "run", "analyze"} and args.video is None:
        parser.error("--video is required")
    if args.action in {"run", "analyze"} and args.packet is None:
        parser.error("--packet is required")
    if args.action in {"collect", "analyze"} and args.results is None:
        parser.error("--results is required")
    if args.action == "bundle":
        from . import package

        result = package.bundle(args.output)
    elif args.action == "prepare":
        from . import runtime

        manifest = runtime.prepare(args.volume)
        result = {"verified": True, "files": len(manifest["files"])}
    elif args.action == "start":
        from . import runtime

        result = runtime.start(args.volume)
    elif args.action == "preflight":
        import os

        from tools.vlm_benchmark import modes
        from tools.vlm_benchmark.lock import acquire

        from . import benchmark, runtime

        runtime.verify_active(args.volume)
        lock_root = modes.ROOT / "outputs" / "vlm-modes" / "run"
        lock_root.parent.parent.mkdir(parents=True, exist_ok=True)
        lock = acquire(lock_root)
        try:
            result = benchmark.preflight(args.video)
        finally:
            os.close(lock)
    elif args.action == "run":
        from . import benchmark, runtime

        active = runtime.verify_active(args.volume)
        deployment = json.loads(Path(active["deployment"]).read_text())
        result = benchmark.run(
            args.video,
            args.output,
            args.packet,
            deployment=deployment,
        )
    elif args.action == "analyze":
        from tools.vlm_benchmark.model_comparison import load_reference

        from .report import analyze

        result = analyze(args.results, load_reference(args.packet, args.video))
    else:
        from . import package

        result = package.collect(args.results, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
