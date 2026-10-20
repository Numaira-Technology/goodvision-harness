"""Explicit FP8 concurrency experiments; no restart or inference by default."""

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from . import fp8_profile as base
from . import runtime


@dataclass(frozen=True)
class Experiment:
    sequences: int = 16
    admission: str = "matched"
    kernel: str = "auto"
    tuned_dir: str | None = None

    def __post_init__(self):
        if type(self.sequences) is not int or self.sequences not in (16, 24, 32):
            raise ValueError("engine concurrency must be 16, 24 or 32")
        if self.admission not in ("matched", "fixed48"):
            raise ValueError("admission must be matched or fixed48")
        if self.kernel not in ("auto", "triton-tuned", "marlin"):
            raise ValueError("unsupported kernel experiment")
        if (self.kernel == "triton-tuned") != bool(self.tuned_dir):
            raise ValueError("triton-tuned requires --tuned-dir exclusively")
        if self.tuned_dir and not Path(self.tuned_dir).is_absolute():
            raise ValueError("tuned-dir must be an absolute persistent path")

    @property
    def global_limit(self):
        return self.sequences if self.admission == "matched" else 48

    @property
    def name(self):
        return f"seq{self.sequences}-{self.admission}-{self.kernel}"


def validate_arguments(args):
    experiment = Experiment(**args.fp8_experiment)
    expected = {**base.FROZEN, "global_limit": experiment.global_limit}
    changed = [k for k, v in expected.items() if getattr(args, k, None) != v]
    if changed:
        raise ValueError("FP8 experiment changed frozen load: " + ", ".join(changed))
    base.policy_check()


def arguments(experiment, *, video=None, real=False):
    from tools.vlm_benchmark import modes

    from . import fp8_adapter

    args = fp8_adapter.arguments(video=video, real=real)
    args.fp8_experiment = asdict(experiment)
    args.global_limit = experiment.global_limit
    modes.validate(args)
    return args


def plan(experiment):
    from . import fp8_adapter

    return {
        **fp8_adapter.plan(),
        "experiment": asdict(experiment),
        "experiment_id": experiment.name,
        "global_limit": experiment.global_limit,
        "server_sequences": experiment.sequences,
        "max_num_batched_tokens": 2048,
        "cache_policy": "unchanged: prefix=off;multimodal_processor=off",
        "kernel_status": "candidate; no throughput or correctness claim",
        "report_title": "YOLO26-S + Qwen35B FP8 / " + experiment.name,
        "comparison_notes": [
            "Only engine concurrency/admission and the selected kernel arm vary.",
            "All 828 planned windows per branch/round remain in the denominator.",
            "Matched changes both client admission and engine concurrency; use "
            "fixed48 to isolate engine concurrency. Do not mix them in one sweep.",
            "Keep one kernel arm across 16/24/32; "
            "compare kernels at fixed concurrency.",
            "No automatic fallback, model changes, "
            "output compression or cache changes.",
        ],
    }


def matrix(experiment):
    return {
        "execute": False,
        "variants": [
            plan(Experiment(**{**asdict(experiment), "sequences": n}))
            for n in (16, 24, 32)
        ],
        "formal_branch_windows": 3 * 6624,
        "formal_rounds": 18,
        "automatic_server_restart": False,
    }


def kernel_evidence(experiment, root):
    if experiment.kernel != "triton-tuned":
        return None
    from .fp8_kernels import verify_bundle

    directory = Path(experiment.tuned_dir)
    if not directory.resolve(strict=True).is_relative_to(Path(root).resolve()):
        raise ValueError("tuned kernels must be on the verified model volume")
    return verify_bundle(directory)


def deployment_profile(experiment):
    backend = {"auto": "auto", "triton-tuned": "triton", "marlin": "marlin"}[
        experiment.kernel
    ]

    def extra(root):
        return {
            "experiment": asdict(experiment),
            "kernel_bundle": kernel_evidence(experiment, root),
        }

    def environment(root):
        evidence = kernel_evidence(experiment, root)
        if evidence is None:
            return {}
        import importlib.metadata

        if evidence["triton_version"] != importlib.metadata.version("triton"):
            raise ValueError("tuned kernel Triton version mismatch")
        return {"VLLM_TUNED_CONFIG_FOLDER": str(Path(experiment.tuned_dir).resolve())}

    def attest(run):
        text = (run / "server.log").read_text()
        selected = [
            line.strip()
            for line in text.splitlines()
            if "Selected " in line or "Fp8 MoE backend" in line
        ]
        actual_backends = set(re.findall(r"Using (\w+) Fp8 MoE backend", text))
        if len(actual_backends) != 1:
            raise ValueError("startup did not attest actual FP8 MoE backend")
        if backend != "auto" and actual_backends != {backend.upper()}:
            raise ValueError("requested FP8 kernel was not selected; no fallback")
        if experiment.kernel == "triton-tuned":
            from .fp8_kernels import CONFIG_NAME

            expected_file = Path(experiment.tuned_dir).resolve() / CONFIG_NAME
            marker = f"Using configuration from {expected_file} for MoE layer."
            if "Using default MoE config" in text or marker not in text:
                raise ValueError("tuned kernel load unconfirmed; no default fallback")
        data = json.loads((run / "deployment.json").read_text())
        data["kernel_attestation"] = {
            "selected": selected,
            "moe_backend": next(iter(actual_backends)),
            "validated": True,
            "capacity_validated": False,
        }
        runtime.write_json(run / "deployment.json", data)

    profile = SimpleNamespace(
        **{k: getattr(base, k) for k in dir(base) if k.isupper() or k == "validate_gpu"}
    )
    profile.SERVER_SEQUENCES = experiment.sequences
    profile.SERVER_OPTIONS = (
        "--max-num-batched-tokens",
        "2048",
        "--enable-chunked-prefill",
        "--moe-backend",
        backend,
    )
    profile.deployment_extra = extra
    profile.server_environment = environment
    profile.attest_ready = attest

    def verify(root, active):
        data = json.loads(Path(active["deployment"]).read_text())
        expected = runtime.serve_command(root, p=profile)
        # The serving interpreter differs from the isolated client interpreter.
        actual = Path(f"/proc/{active['pid']}/cmdline").read_bytes().split(b"\0")
        actual = [s.decode() for s in actual if s]
        if experiment.kernel == "triton-tuned":
            process_env = (
                Path(f"/proc/{active['pid']}/environ").read_bytes().split(b"\0")
            )
            required = (
                "VLLM_TUNED_CONFIG_FOLDER=" + str(Path(experiment.tuned_dir).resolve())
            ).encode()
            if required not in process_env:
                raise ValueError("live server tuned kernel environment mismatch")
        if (
            data.get("experiment") != asdict(experiment)
            or data.get("command", [])[1:] != expected[1:]
            or actual[1:] != expected[1:]
            or data.get("kernel_bundle") != kernel_evidence(experiment, root)
            or not data.get("kernel_attestation", {}).get("validated")
        ):
            raise ValueError("live server does not match requested FP8 experiment")

    profile.verify_deployment = verify
    return profile


class Adapter:
    def __init__(self, experiment):
        self.experiment = experiment

    def arguments(self, **kwargs):
        return arguments(self.experiment, **kwargs)

    def plan(self):
        return plan(self.experiment)

    @staticmethod
    def inspector(*args, **kwargs):
        from .fp8_adapter import inspector

        return inspector(*args, **kwargs)

    @staticmethod
    def stage_metrics(*args, **kwargs):
        from .fp8_adapter import stage_metrics

        return stage_metrics(*args, **kwargs)


def run(experiment, video, output, packet, volume, *, preflight_only=False):
    from tools.tracked_grounding import suite

    from . import benchmark, yolo_runtime

    if Path(output).exists():
        raise ValueError("choose a new output directory")
    active = runtime.verify_active(volume, p=deployment_profile(experiment))
    deployment = {
        "vlm": json.loads(Path(active["deployment"]).read_text()),
        "detector": yolo_runtime.verify_active(volume),
    }
    benchmark.readiness()
    try:
        return suite.run(
            video,
            output,
            base.DETECTOR_URL,
            packet,
            model_adapter=Adapter(experiment),
            preflight_only=preflight_only,
        )
    finally:
        if Path(output).is_dir():
            runtime.write_json(Path(output) / "deployment.json", deployment)


def compare(roots, reference):
    """Recompute saved trials; never use an incomplete arm as a winning score."""
    from tools.tracked_grounding.report import summarize_job
    from tools.vlm_benchmark.accounting import complete
    from tools.vlm_benchmark.model_comparison import safe_path
    from tools.vlm_multistream_benchmark import report_lane

    from .fp8_adapter import stage_metrics
    from .profile import POLICY_SHA256, VIDEO_SHA256

    variants, identities, concurrency = [], set(), set()
    for root in map(Path, roots):
        manifest = json.loads((root / "manifest.json").read_text())
        summary = json.loads((root / "summary.json").read_text())
        deployment = json.loads((root / "deployment.json").read_text())["vlm"]
        exp = Experiment(**manifest["experiment"])
        saved_settings = dict(manifest["settings"])
        if manifest.get("version") == "vast-qwen35-fp8-yolo26s-v1":
            # Read-only migration: v1 coupled backend ownership to this display
            # flag. Preserve the original manifest and its measured results.
            saved_settings.setdefault(
                "backend_person_tracks", saved_settings.get("person_ids_only")
            )
        validate_arguments(SimpleNamespace(**saved_settings))
        expected_plan = plan(exp)
        identity_fields = (
            "model",
            "revision",
            "image",
            "quantization",
            "detector",
            "yolo_weights_sha256",
            "server_sequences",
            "global_limit",
            "max_num_batched_tokens",
            "cache_policy",
            "gpu_memory_utilization",
        )
        if any(
            document.get(k) != expected_plan[k]
            for document in (manifest, summary)
            for k in identity_fields
        ):
            raise ValueError("model/detector/engine configuration changed")
        if (
            manifest["settings"].get("real") is not True
            or manifest.get("preflight_only") is not False
            or manifest["settings"]["fp8_experiment"] != asdict(exp)
            or summary.get("experiment") != asdict(exp)
            or deployment.get("experiment") != asdict(exp)
            or not deployment.get("kernel_attestation", {}).get("validated")
            or manifest["source_sha256"] != VIDEO_SHA256
            or manifest["event_policy"]["sha256"] != POLICY_SHA256
            or reference["policy"]["sha256"] != POLICY_SHA256
            or manifest["decoded_evidence"] != reference["decoded_evidence"]
            or summary["state"] != "completed"
        ):
            raise ValueError("unverified or incomplete comparison arm")
        if any(
            deployment.get(k) != v
            for k, v in {
                "model": base.MODEL_REPO,
                "revision": base.MODEL_REVISION,
                "vllm_version": base.ENGINE_VERSION,
                "image_expected": base.IMAGE,
            }.items()
        ):
            raise ValueError("deployment model or engine differs")
        if not manifest.get("source_hashes"):
            raise ValueError("missing source provenance")
        if exp.sequences in concurrency:
            raise ValueError("duplicate concurrency arm")
        concurrency.add(exp.sequences)
        identities.add(
            (
                exp.admission,
                exp.kernel,
                json.dumps(deployment.get("kernel_bundle"), sort_keys=True),
                deployment["kernel_attestation"]["moe_backend"],
                json.dumps(manifest["source_hashes"], sort_keys=True),
            )
        )
        trials, seen = [], set()
        for job in summary["jobs"]:
            if job["phase"] != "measurement":
                continue
            key = (job["mode"], job["repeat"])
            if (
                key in seen
                or job["streams"] != 12
                or job["duration"] != base.FROZEN["duration"]
            ):
                raise ValueError("duplicate trial or wrong stream count")
            seen.add(key)
            raw = [
                json.loads(line)
                for line in safe_path(root, job["raw_reports"]).read_text().splitlines()
            ]
            expected_lanes = {
                "events": {"events"},
                "compact": {"grounding"},
                "parallel": {"events", "grounding"},
            }.get(job["mode"])
            if set(job["result"]["lanes"]) != expected_lanes:
                raise ValueError("missing or unexpected branches")
            windows = [
                (r["camera"], report_lane(r["report"]), r["report"]["window_id"])
                for r in raw
            ]
            if (
                len(set(windows)) != len(windows)
                or any(
                    c not in range(1, 13) or lane not in expected_lanes
                    for c, lane, _ in windows
                )
                or any(
                    n > 69
                    for n in Counter((c, lane) for c, lane, _ in windows).values()
                )
            ):
                raise ValueError("duplicate or out-of-load raw windows")
            if any(
                row["report"]["event_policy"]["sha256"] != POLICY_SHA256 for row in raw
            ):
                raise ValueError("raw result policy changed")
            if any(
                lane["expected_observations"] != 828
                for lane in job["result"]["lanes"].values()
            ):
                raise ValueError("planned window denominator changed")
            for lane, counts in job["result"]["lanes"].items():
                records = [r["report"] for r in raw if report_lane(r["report"]) == lane]
                if counts["reports"] != len(records) or counts[
                    "complete_valid_results"
                ] != sum(map(complete, records)):
                    raise ValueError(
                        "saved summary does not reconcile with raw results"
                    )
            result = summarize_job(job, raw, reference)
            if not result["valid"]:
                raise ValueError("invalid input/delivery/recording in comparison")
            for lane, data in result["lanes"].items():
                data["tokens"] = stage_metrics(
                    [r["report"] for r in raw if report_lane(r["report"]) == lane],
                    job["result"],
                )
            result["telemetry"] = job.get("telemetry")
            trials.append(result)
        if seen != {(m, n) for m in ("events", "compact", "parallel") for n in (1, 2)}:
            raise ValueError("comparison requires all six formal trials")
        variants.append({"experiment": asdict(exp), "trials": trials})
    if concurrency != {16, 24, 32} or len(identities) != 1:
        raise ValueError("compare requires 16/24/32 with one admission and kernel arm")
    return {
        "variants": sorted(variants, key=lambda r: r["experiment"]["sequences"]),
        "quality_scope": "Astra scene agreement, not person association accuracy",
        "denominator": "828 per branch per round; missing/failed remain counted",
        "reference_sha256": reference["reference_sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["plan", "start", "preflight", "run", "compare"]
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sequences", type=int, choices=[16, 24, 32], default=16)
    parser.add_argument(
        "--admission", choices=["matched", "fixed48"], default="matched"
    )
    parser.add_argument(
        "--kernel", choices=["auto", "triton-tuned", "marlin"], default="auto"
    )
    parser.add_argument("--tuned-dir")
    parser.add_argument("--volume", type=Path, default=Path("/models"))
    for name in ("video", "packet", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--results", type=Path, nargs=3)
    args = parser.parse_args(argv)
    experiment = Experiment(args.sequences, args.admission, args.kernel, args.tuned_dir)
    if not args.execute or args.action == "plan":
        print(json.dumps(matrix(experiment), indent=2))
        return 0
    if args.action == "start":
        from .yolo_runtime import verify_active

        verify_active(args.volume)
        runtime.start(args.volume, p=deployment_profile(experiment))
    elif args.action == "compare":
        from tools.vlm_benchmark.model_comparison import load_reference

        if not all((args.video, args.packet, args.output, args.results)):
            parser.error("--video --packet --output --results required")
        if args.output.exists():
            raise ValueError("comparison output already exists")
        result = compare(args.results, load_reference(args.packet, args.video))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        runtime.write_json(args.output, result)
    else:
        if not all((args.video, args.packet, args.output)):
            parser.error("--video --packet --output required")
        run(
            experiment,
            args.video,
            args.output,
            args.packet,
            args.volume,
            preflight_only=args.action == "preflight",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
