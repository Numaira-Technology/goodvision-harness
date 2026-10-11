"""Twelve-stream grounding 4/6/8 comparison; simulated unless --real is explicit."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.vlm_benchmark import modes  # noqa: E402
from tools.vlm_benchmark.routes import KEY_ENV, read_api_key  # noqa: E402


def parser():
    p = modes.parser()
    p.description = __doc__
    p.set_defaults(
        modes=["grounding"], streams=[12], grounding_format="lean-v1",
        grounding_concurrency_sweep=[4, 6, 8], duration=55.798021,
        warmup_seconds=10, repeats=2, http_timeout=20,
        global_limit=48, per_stream_limit=4, local_concurrency=4,
    )
    p.add_argument("--env-file", type=Path, default=ROOT / ".env",
                   help="read selected API key locally; never save its value")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        modes.validate(args)
    except ValueError as exc:
        p.error(str(exc))
    print(json.dumps({"request_budget": modes.request_budget(args),
                      "real_model": args.real, "stages": list(modes.jobs(args))}))
    if args.plan_only:
        return
    if args.real and args.backend in KEY_ENV:
        os.environ[KEY_ENV[args.backend]] = read_api_key(args.backend, args.env_file)
    output = modes.run(args)
    print(f"Concurrency report: {output / 'grounding-concurrency.html'}")


if __name__ == "__main__":
    main()
