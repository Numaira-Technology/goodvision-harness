"""All GPU/model work requires an explicit subcommand; prepare is offline."""

import argparse
import json
from pathlib import Path

from .contract import MODEL, REVISION, plan
from .packet import prepare
from .ppe import ORDERS, VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("prepare")
    p.add_argument("--source-packet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--windows", type=int, default=12)
    p = sub.add_parser("download")
    p.add_argument("--cache-dir", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--scope", choices=("scene", "person"), default="scene")
    p.add_argument("--detector-url")
    p.add_argument("--person-run", type=Path)
    p.add_argument("--ppe-variant", choices=VARIANTS, default="baseline")
    p.add_argument("--ppe-option-order", choices=ORDERS, default="canonical")
    p = sub.add_parser(
        "preview", help="write actual requests offline; no model or network"
    )
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--person-run", type=Path)
    p.add_argument("--ppe-variant", choices=VARIANTS, default="baseline")
    p.add_argument("--ppe-option-order", choices=ORDERS, default="canonical")
    args = parser.parse_args()
    if args.command == "plan":
        result = plan()
    elif args.command == "prepare":
        result = {
            "packet": str(prepare(args.source_packet, args.output, args.windows)),
            "state": "prepared_not_model_tested",
        }
    elif args.command == "preview":
        from .preview import preview

        result = preview(
            args.packet,
            args.output,
            ppe_variant=args.ppe_variant,
            ppe_option_order=args.ppe_option_order,
            person_run=args.person_run,
        )
    elif args.command == "download":
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            MODEL, revision=REVISION, cache_dir=str(args.cache_dir)
        )
        result = {"model": MODEL, "revision": REVISION, "path": path}
    else:
        from .run import execute

        result = execute(
            args.packet,
            args.output,
            args.cache_dir,
            scope=args.scope,
            detector_url=args.detector_url,
            person_run=args.person_run,
            ppe_variant=args.ppe_variant,
            ppe_option_order=args.ppe_option_order,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
