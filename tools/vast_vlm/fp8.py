"""48GB 4090 entrypoint; server lifecycle does not import client dependencies."""

import argparse
import json
from pathlib import Path

from . import fp8_profile as p
from . import runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "plan",
            "prepare",
            "start",
            "start-detector",
            "preflight",
            "run",
            "bundle",
            "collect",
        ],
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--person-display", choices=["ids", "boxes", "off"], default="ids"
    )
    parser.add_argument("--volume", type=Path, default=Path("/models"))
    parser.add_argument(
        "--detector-python", type=Path, default=Path(".venv-vast-yolo/bin/python")
    )
    for name in ("video", "output", "packet", "results"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if not args.execute or args.action == "plan":
        from .fp8_adapter import plan

        print(json.dumps({**plan(), "person_display": args.person_display}, indent=2))
        return 0
    if args.action in {"run", "preflight"} and not all(
        (args.video, args.output, args.packet)
    ):
        parser.error("--video --output --packet required")
    if args.action == "bundle" and not args.output:
        parser.error("--output required")
    if args.action == "collect" and not all((args.results, args.output)):
        parser.error("--results --output required")
    if args.action == "collect":
        from .package import collect

        print(json.dumps(collect(args.results, args.output)))
    elif args.action == "prepare":
        from .yolo_runtime import prepare

        runtime.prepare(args.volume, p=p)
        prepare(args.volume)
    elif args.action == "start":
        from .yolo_runtime import verify_active

        verify_active(args.volume)  # Load and warm YOLO before reserving VLM memory.
        runtime.start(args.volume, p=p)
    elif args.action == "start-detector":
        from .yolo_runtime import start

        start(args.volume, args.detector_python)
    elif args.action == "bundle":
        from .package import bundle

        print(json.dumps(bundle(args.output)))
    else:
        from .fp8_adapter import run

        run(
            args.video,
            args.output,
            args.packet,
            args.volume,
            preflight_only=args.action == "preflight",
            person_display=args.person_display,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
