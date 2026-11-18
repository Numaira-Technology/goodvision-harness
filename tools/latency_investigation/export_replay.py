"""Export a stratified slice of recorded Qwen chat requests for standard tools.

Primary dataset: vLLM 0.31.0 ``custom_image`` JSONL (``prompt`` + ``image_files``).
Secondary dataset: AIPerf 0.13.0 ``single_turn`` JSONL (``text``, ``image``,
``timestamp`` in milliseconds, ``output_length``, ``extra.temperature``).

The replay engines are those tools. This module only writes the files, the
manifest, and the command plan. It does not send requests and does not rent
a GPU.

Closed-loop latency is vLLM bench serve's E2EL, which starts at actual send
after the concurrency semaphore. Open-loop replay is AIPerf ``--fixed-schedule``;
that tool's ``request_latency`` starts when the request is dispatched, and its
schedule-lag metrics are separate. See the package README.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import random
from pathlib import Path

from tools.latency_investigation.percentiles import token_bucket

VLLM_VERSION = "0.31.0"
AIPERF_VERSION = "0.13.0"
MODEL_NAME = "qwen/qwen3-vl-8b-instruct"
DEFAULT_SIZE = 250
DEFAULT_SEED = 20261112
MIN_SIZE = 200
MAX_SIZE = 300
WARMUP_COUNT = 12

CLOCK_NOTE = (
    "Standard-method variant: production replay of the fixed recorded set. "
    "Primary closed loop is `vllm bench serve` "
    f"{VLLM_VERSION} custom_image with --request-rate inf --max-concurrency 12. "
    "Its E2EL (`latencies`) starts at actual HTTP send, after the client "
    "concurrency semaphore; slot wait is `client_queue_time` and is not added "
    "when the request rate is infinite. "
    "Secondary control is AIPerf "
    f"{AIPERF_VERSION} single_turn --fixed-schedule. "
    "Its request_latency starts at dispatch. This report adds the least-late "
    "schedule lag, (start - scheduled) minus the minimum of that offset, so "
    "the primary P50/P95 start at the scheduled send. The from-actual-send "
    "P50/P95 stay in the secondary columns. "
    "Neither command sets ignore_eos. max_tokens is the recorded cap "
    "(vLLM sends it as max_completion_tokens) and generation stops naturally. "
    "Weight download and env restore are not part of these latencies."
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def image_bytes_from_url(url: str) -> bytes:
    prefix = "base64,"
    if not isinstance(url, str) or prefix not in url:
        raise ValueError("image url must be a base64 data url")
    payload = url.split(prefix, 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("image data url is not valid base64") from error


def text_and_image_from_body(body: dict) -> tuple[str, bytes]:
    messages = body.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or not isinstance(messages[0], dict)
    ):
        raise ValueError("chat body needs one user message")
    content = messages[0].get("content")
    if not isinstance(content, list):
        raise ValueError("chat content must be a list of parts")
    text = None
    image = None
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            text = part["text"]
        if part.get("type") == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if isinstance(url, str):
                image = image_bytes_from_url(url)
    if not isinstance(text, str) or image is None:
        raise ValueError("chat body needs text and one image")
    return text, image


def normalize_record(record: dict) -> dict:
    """Accept either a chat completion body or an already split record."""
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id is required")
    prompt_tokens = record.get("prompt_tokens")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
        raise ValueError("prompt_tokens must be an int")
    body = record.get("body")
    if isinstance(body, dict):
        text, image = text_and_image_from_body(body)
        max_tokens = body.get("max_tokens", record.get("max_tokens"))
        temperature = body.get("temperature", record.get("temperature"))
    else:
        text = record.get("text")
        image = record.get("image_bytes")
        if isinstance(record.get("image_path"), str):
            image = Path(record["image_path"]).read_bytes()
        max_tokens = record.get("max_tokens")
        temperature = record.get("temperature")
    if not isinstance(text, str) or not isinstance(image, (bytes, bytearray)):
        raise ValueError("text and image bytes are required")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise ValueError("max_tokens must be a positive int")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a number")
    if not math.isfinite(float(temperature)):
        raise ValueError("temperature must be finite")
    arrival = record.get("arrival_offset_seconds")
    if arrival is not None and (
        isinstance(arrival, bool)
        or not isinstance(arrival, (int, float))
        or not math.isfinite(float(arrival))
        or arrival < 0
    ):
        raise ValueError("arrival_offset_seconds must be a non-negative number")
    return {
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "bucket": token_bucket(prompt_tokens),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "text": text,
        "image_bytes": bytes(image),
        "arrival_offset_seconds": None if arrival is None else float(arrival),
        "output_text": record.get("output_text")
        if isinstance(record.get("output_text"), str)
        else None,
        "output_tokens": record.get("output_tokens")
        if isinstance(record.get("output_tokens"), int)
        and not isinstance(record.get("output_tokens"), bool)
        else None,
    }


def choose_subset(records, *, size=DEFAULT_SIZE, seed=DEFAULT_SEED):
    """Stratify without replacement so the long-request share matches the source.

    ``size`` is clamped to 200..300 when the population is at least 200.
    Smaller populations are returned whole. The 500-999 share of the sample
    matches the source share to the nearest request.
    """
    rows = list(records)
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate request_id")
    if not rows:
        raise ValueError("no records")
    target = size
    if len(rows) >= MIN_SIZE:
        target = min(MAX_SIZE, max(MIN_SIZE, size))
        target = min(target, len(rows))
    else:
        target = len(rows)
    long_rows = [row for row in rows if row["bucket"] == "500-999"]
    other_rows = [row for row in rows if row["bucket"] != "500-999"]
    long_n = round(target * (len(long_rows) / len(rows)))
    long_n = min(long_n, len(long_rows), target)
    other_n = target - long_n
    if other_n > len(other_rows):
        long_n += other_n - len(other_rows)
        other_n = len(other_rows)
        long_n = min(long_n, len(long_rows))
    rng = random.Random(seed)
    long_ids = [row["request_id"] for row in long_rows]
    other_ids = [row["request_id"] for row in other_rows]
    rng.shuffle(long_ids)
    rng.shuffle(other_ids)
    chosen = set(long_ids[:long_n] + other_ids[:other_n])
    # Stable order is the source order, not the shuffle, so the file is reproducible.
    selected = [row for row in rows if row["request_id"] in chosen]
    warmup_pool = [row for row in rows if row["request_id"] not in chosen]
    warmup = warmup_pool[:WARMUP_COUNT]
    return selected, warmup


def _write_images(directory: Path, rows, prefix: str) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for index, row in enumerate(rows):
        name = f"{prefix}{index:04d}.jpg"
        path = directory / name
        path.write_bytes(row["image_bytes"])
        written.append({**row, "image_rel": str(Path(directory.name) / name)})
    return written


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _shared_generation(rows):
    tokens = {row["max_tokens"] for row in rows}
    temperatures = {row["temperature"] for row in rows}
    if len(tokens) != 1 or len(temperatures) != 1:
        raise ValueError("replay set must share one max_tokens and one temperature")
    return tokens.pop(), temperatures.pop()


def export_subset(
    records, destination, *, size=DEFAULT_SIZE, seed=DEFAULT_SEED
) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"{destination} already exists")
    normalized = [normalize_record(record) for record in records]
    selected, warmup = choose_subset(normalized, size=size, seed=seed)
    max_tokens, temperature = _shared_generation(selected)
    destination.mkdir(parents=True)
    timed = _write_images(destination / "images", selected, "")
    warm = _write_images(destination / "warmup-images", warmup, "")
    vllm_rows = [
        {"prompt": row["text"], "image_files": [row["image_rel"]]} for row in timed
    ]
    aiperf_rows = []
    for index, row in enumerate(timed):
        item = {
            "session_id": row["request_id"],
            "text": row["text"],
            "image": row["image_rel"],
            "output_length": row["max_tokens"],
            "extra": {"temperature": row["temperature"]},
        }
        if row["arrival_offset_seconds"] is not None:
            # AIPerf single_turn timestamps are milliseconds.
            item["timestamp"] = round(row["arrival_offset_seconds"] * 1000.0, 3)
        elif timed:
            item["timestamp"] = index
        aiperf_rows.append(item)
    _jsonl(destination / "vllm-custom-image.jsonl", vllm_rows)
    _jsonl(destination / "aiperf-single-turn.jsonl", aiperf_rows)
    warm_vllm = [
        {"prompt": row["text"], "image_files": [row["image_rel"]]} for row in warm
    ]
    _jsonl(destination / "warmup-vllm-custom-image.jsonl", warm_vllm)
    commands = command_plan(
        destination,
        max_tokens=max_tokens,
        temperature=temperature,
        timed_count=len(timed),
        warmup_count=len(warm),
    )
    (destination / "commands.json").write_text(
        json.dumps(commands, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    identities = []
    for row in timed:
        identities.append(
            {
                "request_id": row["request_id"],
                "prompt_tokens": row["prompt_tokens"],
                "bucket": row["bucket"],
                "max_tokens": row["max_tokens"],
                "temperature": row["temperature"],
                "arrival_offset_seconds": row["arrival_offset_seconds"],
                "image": row["image_rel"],
                "image_sha256": _sha256(row["image_bytes"]),
                "baseline_output_text": row["output_text"],
                "baseline_output_tokens": row["output_tokens"],
            }
        )
    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            files[path.relative_to(destination).as_posix()] = _file_sha256(path)
    manifest = {
        "schema": "goodvision.latency_replay.v1",
        "vllm_version": VLLM_VERSION,
        "aiperf_version": AIPERF_VERSION,
        "seed": seed,
        "timed_count": len(timed),
        "warmup_count": len(warm),
        "warmup_request_ids": [row["request_id"] for row in warm],
        "timed_request_ids": [row["request_id"] for row in timed],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "source_count": len(normalized),
        "source_500_999": sum(1 for row in normalized if row["bucket"] == "500-999"),
        "timed_500_999": sum(1 for row in timed if row["bucket"] == "500-999"),
        "requests": identities,
        "files_sha256": files,
        "clock": CLOCK_NOTE,
        "warmup_rule": (
            "Warmup requests are drawn from the complement of the timed set. "
            "Restart vLLM, run the warmup file, then start the timed round. "
            "Do not include warmup latencies in P50/P95."
        ),
    }
    # The manifest hashes every other file, then records its own hash beside it.
    manifest_path = destination / "manifest.json"
    encoded = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest_path.write_bytes(encoded)
    digest = _sha256(encoded)
    (destination / "manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    return manifest


def command_plan(
    destination, *, max_tokens, temperature, timed_count, warmup_count
) -> dict:
    destination = Path(destination).resolve()
    common = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--model",
        MODEL_NAME,
        "--endpoint",
        "/v1/chat/completions",
        "--dataset-name",
        "custom_image",
        "--disable-shuffle",
        "--no-oversample",
        "--custom-ensure-client-side-data",
        "--custom-output-len",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--request-rate",
        "inf",
        "--max-concurrency",
        "12",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(destination / "results"),
        "--metric-percentiles",
        "50,95",
        "--percentile-metrics",
        "ttft,e2el",
    ]
    plan = {
        "tool_versions": {"vllm": VLLM_VERSION, "aiperf": AIPERF_VERSION},
        "do_not_run_until_authorized": True,
        "rents_gpu": False,
        "cwd": str(destination),
        "clock": CLOCK_NOTE,
        "steps": [
            {
                "name": "restart-vllm",
                "action": (
                    "Restart the already running vLLM 0.31.0 server with the "
                    "existing tools.qwen8b.serve_vllm command so prefix and "
                    "multimodal caches are empty. This plan does not rent a GPU."
                ),
            },
            {
                "name": "warmup",
                "excluded_from_percentiles": True,
                "request_count": warmup_count,
                "argv": [
                    *common,
                    "--dataset-path",
                    "warmup-vllm-custom-image.jsonl",
                    "--num-prompts",
                    str(warmup_count),
                    "--result-filename",
                    "warmup.json",
                    "--label",
                    "warmup",
                ],
            },
            {
                "name": "closed-loop",
                "latency_starts": "actual_send",
                "argv": [
                    *common,
                    "--dataset-path",
                    "vllm-custom-image.jsonl",
                    "--num-prompts",
                    str(timed_count),
                    "--result-filename",
                    "closed-loop.json",
                    "--label",
                    "closed-loop",
                ],
            },
            {
                "name": "open-loop",
                "latency_starts": "scheduled_send",
                "actual_send_column": "p50_actual_send_seconds",
                "argv": [
                    "aiperf",
                    "profile",
                    "--model",
                    MODEL_NAME,
                    "--endpoint-type",
                    "chat",
                    "--url",
                    "localhost:8000",
                    "--input-file",
                    "aiperf-single-turn.jsonl",
                    "--custom-dataset-type",
                    "single_turn",
                    "--streaming",
                    "--fixed-schedule",
                    "--concurrency",
                    "12",
                    "--request-count",
                    str(timed_count),
                    "--output-artifact-dir",
                    str(destination / "aiperf"),
                ],
            },
        ],
    }

    steps = {step["name"]: step for step in plan["steps"]}

    def metrics_step(run, when):
        return {
            "name": f"{run}-metrics-{when}",
            "argv": [
                "curl",
                "-fsS",
                "http://127.0.0.1:8000/metrics",
                "-o",
                f"{run}-metrics-{when}.prom",
            ],
        }

    plan["steps"] = [steps["restart-vllm"]]
    if warmup_count:
        plan["steps"].append(steps["warmup"])
    for run in ("closed-loop", "open-loop"):
        if run == "open-loop":
            plan["steps"].append(
                {**steps["restart-vllm"], "name": "open-loop-restart-vllm"}
            )
            if warmup_count:
                warmup = {**steps["warmup"], "name": "open-loop-warmup"}
                warmup["argv"] = [
                    "open-loop-warmup.json" if arg == "warmup.json" else arg
                    for arg in warmup["argv"]
                ]
                plan["steps"].append(warmup)
        plan["steps"].extend(
            [metrics_step(run, "before"), steps[run], metrics_step(run, "after")]
        )
    return plan


def records_from_phase(bundle, phase_dir, alias: str) -> list[dict]:
    """Rebuild recorded chat bodies for one saved eval phase.

    Arrival offsets are seconds after the first intent in that phase.
    Prompt tokens and output text come from the saved receipt. The formal
    elapsed time is not copied into the replay body.
    """
    from tools.unified_benchmark.adapters import request_body
    from tools.unified_benchmark.contract import digest, finite_number
    from tools.unified_benchmark.http_execution import (
        _verify_completion, receipt_path, reconcile,
    )
    from tools.unified_benchmark.receipts import read_receipt
    from tools.unified_benchmark.retry import Ledger
    from tools.unified_benchmark.workflow import context_for, verify_bundle

    bundle = Path(bundle)
    phase_dir = Path(phase_dir)
    if not alias or Path(alias).name != alias or alias in {".", ".."}:
        raise ValueError("invalid replay alias")
    if phase_dir.name not in {"validation", "test"} or phase_dir.is_symlink():
        raise ValueError("invalid replay phase")
    bundle_digest = verify_bundle(bundle)
    config = json.loads(
        (bundle / "configs" / f"{alias}.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (bundle / "plans" / f"{alias}-{phase_dir.name}.json").read_text(
            encoding="utf-8"
        )
    )
    units = {unit["id"]: unit for unit in plan["units"]}
    bodies = {}
    for unit_id, unit in units.items():
        body = request_body(config, unit, bundle)[1]
        if digest(body) != unit["request_sha256"]:
            raise ValueError("recorded replay request fingerprint differs")
        bodies[unit_id] = body
    output = phase_dir.parent
    manifest = output / "run.json"
    ledger_path = phase_dir / "attempts.jsonl"
    if manifest.is_symlink() or ledger_path.is_symlink() or not ledger_path.is_file():
        raise ValueError("invalid replay evidence path")
    identity = json.loads(manifest.read_text(encoding="utf-8"))
    if any(identity.get(key) != expected for key, expected in {
        "alias": alias, "bundle_sha256": bundle_digest, "config_sha256": digest(config),
    }.items()) or identity.get("phase") not in {phase_dir.name, "both"}:
        raise ValueError("replay run identity differs")
    ledger = Ledger(ledger_path, context_for(plan, config, bundle_digest), read_only=True)
    try:
        sealed = _verify_completion(output, phase_dir, ledger, config, phase_dir.name)
        reconcile(phase_dir, ledger, plan, config,
                  local=config.get("execution") == "self_hosted_r2",
                  sealed=sealed, validate_only=True)
        rows = list(ledger.rows)
        receipts = {}
        for row in rows:
            if row["type"] != "result" or row.get("complete") is not True:
                continue
            path = receipt_path(phase_dir, units[row["unit"]], row["attempt"])
            metadata = row.get("metadata", {})
            if (metadata.get("receipt") != path.relative_to(phase_dir).as_posix()
                    or metadata.get("receipt_sha256") != _file_sha256(path)):
                raise ValueError("replay receipt anchor differs")
            receipts[row["unit"], row["attempt"]] = read_receipt(path)
    finally:
        ledger.close()
    intents = {}
    results = {}
    for row in sorted(rows, key=lambda row: row["attempt"]):
        if row.get("type") == "intent":
            if not finite_number(row.get("at_unix")) or row["at_unix"] < 0:
                raise ValueError("invalid replay intent timestamp")
            intents[row["unit"], row["attempt"]] = row
        elif row.get("type") == "result" and row.get("complete") is True:
            results.setdefault(row["unit"], row)
    if not intents:
        raise ValueError("phase has no request intents")
    origin = min(row["at_unix"] for row in intents.values())
    records = []
    for unit_id in (unit["id"] for unit in plan["units"]):
        result = results.get(unit_id)
        if result is None:
            continue
        key = unit_id, result["attempt"]
        intent = intents[key]
        receipt = receipts[key]
        response = (
            receipt.get("response") if isinstance(receipt.get("response"), dict) else {}
        )
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        choices = (
            response.get("choices") if isinstance(response.get("choices"), list) else []
        )
        message = (
            choices[0].get("message")
            if choices and isinstance(choices[0], dict)
            else {}
        )
        output_text = message.get("content") if isinstance(message, dict) else None
        records.append(
            {
                "request_id": unit_id,
                "prompt_tokens": prompt_tokens,
                "arrival_offset_seconds": float(intent["at_unix"]) - origin,
                "output_text": output_text if isinstance(output_text, str) else None,
                "output_tokens": output_tokens
                if isinstance(output_tokens, int)
                else None,
                "body": bodies[unit_id],
            }
        )
    return records
