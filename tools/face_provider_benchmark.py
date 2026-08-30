"""Benchmark and cross-check InsightFace execution providers on one image."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full InsightFace detection and recognition against the same "
            "image with one or more ONNX Runtime providers."
        )
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--model-root", type=Path, default=Path(".models/insightface")
    )
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument(
        "--provider",
        action="append",
        dest="providers",
        help=(
            "Execution provider to test; repeat for A/B. Defaults to CPU and "
            "DirectML when available."
        ),
    )
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json-output", type=Path)
    return parser


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _session_providers(model: Any, configured_provider: str) -> dict[str, list[str]]:
    modules = getattr(model, "models", None)
    if not isinstance(modules, dict):
        raise RuntimeError("InsightFace did not expose its loaded modules")
    required = {"detection", "recognition"}
    if not required.issubset(modules):
        raise RuntimeError("InsightFace did not load detection and recognition")

    sessions: dict[str, list[str]] = {}
    for name in sorted(required):
        session = getattr(modules[name], "session", None)
        get_providers = getattr(session, "get_providers", None)
        providers = list(get_providers()) if callable(get_providers) else []
        if not providers or providers[0] != configured_provider:
            raise RuntimeError(
                f"{name} requested {configured_provider}, actual providers={providers}"
            )
        sessions[name] = providers
    return sessions


def _face_snapshot(faces: list[Any]) -> dict[str, Any] | None:
    if not faces:
        return None
    face = max(
        faces,
        key=lambda item: float(
            max(0.0, item.bbox[2] - item.bbox[0])
            * max(0.0, item.bbox[3] - item.bbox[1])
        ),
    )
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(face, "embedding", None)
        if embedding is not None:
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding = embedding / norm
    return {
        "bbox": np.asarray(face.bbox, dtype=np.float32),
        "embedding": (
            None if embedding is None else np.asarray(embedding, dtype=np.float32)
        ),
    }


def _run_provider(
    *,
    provider: str,
    image: np.ndarray[Any, Any],
    model_root: Path,
    model_name: str,
    det_size: int,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from insightface.app import FaceAnalysis

    load_started = time.perf_counter()
    model = FaceAnalysis(
        name=model_name,
        root=str(model_root),
        allowed_modules=["detection", "recognition"],
        providers=[provider],
    )
    model.prepare(
        ctx_id=-1 if provider == "CPUExecutionProvider" else 0,
        det_size=(det_size, det_size),
    )
    sessions = _session_providers(model, provider)
    load_ms = (time.perf_counter() - load_started) * 1000.0

    faces: list[Any] = []
    for _ in range(warmup):
        faces = model.get(image)

    latencies: list[float] = []
    face_counts: list[int] = []
    for _ in range(iterations):
        started = time.perf_counter()
        faces = model.get(image)
        latencies.append((time.perf_counter() - started) * 1000.0)
        face_counts.append(len(faces))

    mean_ms = statistics.fmean(latencies)
    result = {
        "provider": provider,
        "actual_sessions": sessions,
        "load_ms": round(load_ms, 3),
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "face_counts": sorted(set(face_counts)),
        "latency_ms": {
            "min": round(min(latencies), 3),
            "mean": round(mean_ms, 3),
            "p50": round(_percentile(latencies, 0.50), 3),
            "p90": round(_percentile(latencies, 0.90), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3),
        },
        "throughput_fps": round(1000.0 / mean_ms, 3),
    }
    return result, _face_snapshot(faces)


def _cross_check(
    results: list[dict[str, Any]], snapshots: list[dict[str, Any] | None]
) -> dict[str, Any] | None:
    if len(results) != 2:
        return None
    first, second = snapshots
    output: dict[str, Any] = {
        "providers": [results[0]["provider"], results[1]["provider"]],
        "both_detected_face": first is not None and second is not None,
    }
    if first is None or second is None:
        return output
    output["bbox_max_abs_difference_px"] = round(
        float(np.max(np.abs(first["bbox"] - second["bbox"]))), 4
    )
    first_embedding = first["embedding"]
    second_embedding = second["embedding"]
    if first_embedding is not None and second_embedding is not None:
        output["embedding_cosine_similarity"] = round(
            float(np.dot(first_embedding, second_embedding)), 8
        )
    first_p50 = float(results[0]["latency_ms"]["p50"])
    second_p50 = float(results[1]["latency_ms"]["p50"])
    if first_p50 > 0:
        output["second_provider_p50_change_percent"] = round(
            (second_p50 - first_p50) / first_p50 * 100.0, 2
        )
    return output


def main() -> int:
    args = _parser().parse_args()
    if args.det_size < 1 or args.warmup < 0 or args.iterations < 1:
        raise SystemExit("det-size and iterations must be positive; warmup may be zero")
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")

    import onnxruntime as ort

    available = list(ort.get_available_providers())
    providers = args.providers
    if not providers:
        providers = ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
    unavailable = [provider for provider in providers if provider not in available]
    if unavailable:
        raise SystemExit(
            f"unavailable providers={unavailable}; available providers={available}"
        )

    results: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any] | None] = []
    for provider in providers:
        result, snapshot = _run_provider(
            provider=provider,
            image=image,
            model_root=args.model_root.resolve(),
            model_name=args.model_name,
            det_size=args.det_size,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        results.append(result)
        snapshots.append(snapshot)

    payload = {
        "onnxruntime_version": ort.__version__,
        "available_providers": available,
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "results": results,
        "cross_validation": _cross_check(results, snapshots),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
