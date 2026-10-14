"""Fixed twelve-stream Qwen3.5-35B/Alibaba run; offline unless --real is explicit."""

import argparse
import json
import os
from pathlib import Path

from tools.vlm_benchmark import modes, run_remote
from tools.vlm_benchmark.routes import read_api_key

MODEL = "qwen/qwen3.5-35b-a3b"


def fixed_arguments(*, allow_provider_fallback=False):
    # No model, provider, layout or load overrides in this entry point.
    return [
        "--model",
        MODEL,
        "--provider-prefer" if allow_provider_fallback else "--provider-only",
        "alibaba",
        "--require-route-match",
        "--modes",
        "events",
        "grounding",
        "parallel",
        "--streams",
        "12",
        "--event-policy",
        "research-v1",
        "--grounding-format",
        "lean-v1",
        "--duration",
        "55.798021",
        "--repeats",
        "2",
        "--warmup-seconds",
        "10",
        "--soak-seconds",
        "0",
        "--global-limit",
        "48",
        "--per-stream-limit",
        "4",
        "--request-rate",
        "40",
        "--http-timeout",
        "20",
        "--arrival-mode",
        "staggered",
        "--record",
        "--monitor-layout",
        "single-video",
    ]


def parallel_arguments():
    args = fixed_arguments(allow_provider_fallback=True)
    start, end = args.index("--modes"), args.index("--streams")
    args[start:end] = ["--modes", "parallel"]
    return [*args, "--parallel-concurrency-sweep", "4", "6", "8"]


def main(argv=None, *, parallel_sweep=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--identity", type=Path, default=Path.home() / ".ssh/id_ed25519_vast"
    )
    parser.add_argument(
        "--video", type=Path, default=Path.home() / "Downloads/output 3.mp4"
    )
    parser.add_argument("--real", action="store_true")
    parser.add_argument(
        "--allow-provider-fallback", action="store_true",
        help="prefer Alibaba; permit other providers of the same fixed 35B model",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--cloud", action="store_true", help="prepare a cloud plan without connecting"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume")
    action.add_argument("--stop")
    parser.add_argument("--reuse-video")
    parser.add_argument("--output", type=Path, help="local run output directory")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=modes.ROOT / ".env",
        help="local only; cloud uses existing hidden key entry",
    )
    args = parser.parse_args(argv)
    common = [
        *(parallel_arguments() if parallel_sweep else fixed_arguments(
            allow_provider_fallback=args.allow_provider_fallback
        )),
        "--video", str(args.video),
    ]
    if args.real:
        common.append("--real")
    if args.plan_only:
        common.append("--plan-only")
    if args.host or args.cloud or args.prepare_only:
        if args.output:
            parser.error("cloud runs manage their own output/run IDs")
        extra = ["--suite", "modes", "--provider", "openrouter"]
        for key in ("host", "port", "identity", "resume", "stop", "reuse_video"):
            value = getattr(args, key)
            if value is not None:
                extra += ["--" + key.replace("_", "-"), str(value)]
        if args.prepare_only:
            extra.append("--prepare-only")
        return run_remote.main([*common, *extra])
    if args.port or args.resume or args.stop or args.reuse_video:
        parser.error("SSH lifecycle options require --host")
    options = modes.parser().parse_args([*common, "--backend", "openrouter"])
    options.output = args.output
    options.monitor = True
    modes.validate(options)
    print(
        json.dumps(
            {
                "settings": {
                    k: str(v)
                    if isinstance(v, Path)
                    else [str(p) for p in v]
                    if k == "video"
                    else v
                    for k, v in vars(options).items()
                },
                "request_budget": modes.request_budget(options),
                "jobs": list(modes.jobs(options)),
            },
            indent=2,
        )
    )
    if args.plan_only:
        return 0
    if args.real:
        os.environ["OPENROUTER_API_KEY"] = read_api_key("openrouter", args.env_file)
    modes.run(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
