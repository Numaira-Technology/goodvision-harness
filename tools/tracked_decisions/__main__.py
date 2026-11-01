"""Offline plan by default; --execute is required for real preflight or testing."""

import argparse
import json
from pathlib import Path

from . import suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["plan", "preflight", "run", "bundle", "compare"]
    )
    parser.add_argument("--execute", action="store_true")
    for name in ("video", "packet", "output", "env-file", "results", "baseline"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--detector-url")
    args = parser.parse_args(argv)
    if not args.execute or args.action == "plan":
        print(json.dumps(suite.plan(), ensure_ascii=False, indent=2))
        return 0
    if args.action == "bundle":
        if not args.output:
            parser.error("--output required")
        from tools.decisions_benchmark.package import bundle

        result = bundle(args.output, tracked=True)
    elif args.action == "compare":
        if not all((args.results, args.baseline, args.output, args.packet, args.video)):
            parser.error("--results --baseline --output --packet --video required")
        from .comparison import compare

        result = compare(
            args.results, args.baseline, args.output, args.packet, args.video
        )
    else:
        if not all((args.video, args.packet, args.output, args.detector_url)):
            parser.error("--video --packet --output --detector-url required")
        result = suite.run(
            args.video,
            args.output,
            args.detector_url,
            args.packet,
            env_file=args.env_file,
            preflight_only=args.action == "preflight",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
