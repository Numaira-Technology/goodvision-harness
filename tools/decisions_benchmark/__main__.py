"""Default to offline planning. Explicit --execute enables paid calls."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["plan", "preflight", "run", "analyze", "bundle"]
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args(argv)
    from . import suite

    if args.action == "plan" or not args.execute:
        print(json.dumps(suite.plan(), indent=2))
        return 0
    required = {
        "preflight": ("video", "output"),
        "run": ("video", "packet", "output"),
        "analyze": ("video", "packet", "results"),
        "bundle": ("output",),
    }
    for field in required[args.action]:
        if getattr(args, field) is None:
            parser.error(f"--{field} is required")
    if args.action == "preflight":
        result = suite.preflight(args.video, args.output, env_file=args.env_file)
    elif args.action == "run":
        result = suite.run(args.video, args.output, args.packet, env_file=args.env_file)
    elif args.action == "analyze":
        from tools.vlm_benchmark.model_comparison import load_reference

        from .report import analyze

        result = analyze(args.results, load_reference(args.packet, args.video))
    else:
        from .package import bundle

        result = bundle(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
