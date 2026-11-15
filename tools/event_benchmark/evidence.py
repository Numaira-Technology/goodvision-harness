"""Controlled-run provenance; importing this module never accesses the network."""

import hashlib
import json
import math
import platform
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

VARIANT = "v2-provider-pinned-lossless-v1"
LEGACY_VARIANT = "v2.1-legacy-unpinned-lossless-v1"


def code_provenance():
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, timeout=10, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=root, timeout=10)
    except (OSError, subprocess.SubprocessError):
        commit, diff = None, None
    paths = ["run_eval.py", "tune_threshold.py", "evidence.py", "prepare_provider.py", "SPEC.md", "CONTROLLED_RUN.md"]
    return {"source_commit": commit,
            "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
            "code_file_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in paths if Path(__file__).with_name(name).exists()},
            "client_environment": {"platform": platform.system(), "machine": platform.machine(),
                                   "python": platform.python_version()},
            "provider_hardware": None, "provider_vram_gb": None}


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_provider(path, model, backend):
    config, digest = read_json(path)
    route = config.get("provider", {})
    only = route.get("only")
    if (config.get("schema_version") != 1 or config.get("model") != model
            or config.get("backend") != backend):
        raise ValueError("provider configuration model/backend differs")
    if (not isinstance(only, list) or len(only) != 1
            or not isinstance(only[0], str) or not only[0].strip()
            or route.get("allow_fallbacks") is not False
            or set(route) - {"only", "allow_fallbacks", "require_parameters", "quantizations"}):
        raise ValueError("pin exactly one endpoint with provider fallback disabled")
    if not isinstance(config.get("expected_provider_name"), str) or not config["expected_provider_name"].strip():
        raise ValueError("expected actual provider name is required")
    if not config.get("selection") or not config.get("selected_at"):
        raise ValueError("provider selection evidence and date are required")
    return {"sha256": digest, "config": config}


def provider_body(context):
    # A self-hosted OpenAI-compatible endpoint has no OpenRouter routing block.
    # Keep the frozen chat body provider-free; OpenRouter contexts still inject
    # the pinned provider object unchanged.
    if not context:
        return None
    config = context["config"]
    if config.get("transport") == "selfhost":
        return None
    return dict(config["provider"])


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def response_metadata(payload):
    """Allowlist metadata, never save keys, image payloads or provider error text."""
    if not isinstance(payload, dict):
        return {}
    result = {}
    for source, target in (("id", "generation_id"), ("provider", "actual_provider"),
                           ("provider_name", "actual_provider"), ("model", "actual_model")):
        value = payload.get(source)
        if isinstance(value, str) and value.strip():
            result[target] = value[:512]
    usage = payload.get("usage")
    if isinstance(usage, dict):
        result["usage"] = {key: value for key, value in usage.items()
                           if key in {"prompt_tokens", "completion_tokens", "total_tokens",
                                      "input_tokens", "output_tokens", "cost"}
                           and finite_number(value)}
    for key in ("total_cost", "native_tokens_prompt", "native_tokens_completion"):
        value = payload.get(key)
        if finite_number(value):
            result[key] = value
    return result


def enrich_attempts(attempts, key):
    """GET metadata only, after inference/timing; never repeat an inference call."""
    cache = {}
    for attempt in attempts:
        generation_id = attempt.get("generation_id")
        if not generation_id or (attempt.get("actual_provider") and attempt.get("actual_model")):
            continue
        if generation_id not in cache:
            try:
                request = urllib.request.Request(
                    "https://openrouter.ai/api/v1/generation?id=" + urllib.parse.quote(generation_id, safe=""),
                    headers={"Authorization": "Bearer " + key})
                with urllib.request.urlopen(request, timeout=15) as response:
                    raw = response.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError("oversized metadata")
                data = json.loads(raw)["data"]
                if not isinstance(data, dict):
                    raise ValueError("invalid generation metadata envelope")
                if data.get("id") != generation_id:
                    raise ValueError("metadata generation id differs")
                cache[generation_id] = response_metadata(data)
            except (OSError, ValueError, KeyError, TypeError):
                cache[generation_id] = {"metadata_lookup_error": "unavailable"}
        for field, value in cache[generation_id].items():
            attempt.setdefault(field, value)


def provider_summary(attempts, context):
    expected = context["config"]["expected_provider_name"] if context else None
    providers = sorted({a["actual_provider"] for a in attempts if a.get("actual_provider")})
    models = sorted({a["actual_model"] for a in attempts if a.get("actual_model")})
    successful = [a for a in attempts if a.get("ok")]
    missing = sum(not a.get("actual_provider") or not a.get("actual_model")
                  or not a.get("generation_id") for a in successful)
    return {"actual_providers": providers, "actual_models": models,
            "successful_attempts": len(successful), "successful_attempts_missing_metadata": missing,
            "provider_mismatch": bool(expected and any(p != expected for p in providers)),
            "complete": bool(context and successful and not missing and providers == [expected]
                             and len(models) == 1),
            "actual_endpoint_verified": False, "actual_precision": None,
            "server_weight_revision": None,
            "limits": "Actual provider/model come from receipts. Endpoint, precision and weight revision are not independently attested."}


def verify_provider_evidence(metadata, rows):
    context = metadata.get("provider_context")
    if not isinstance(context, dict) or not context.get("sha256"):
        raise ValueError("pinned provider evidence required")
    config = context.get("config", {})
    route = config.get("provider", {})
    if (config.get("model") != metadata.get("model") or config.get("backend") != metadata.get("backend")
            or route.get("allow_fallbacks") is not False or len(route.get("only", [])) != 1):
        raise ValueError("provider identity or fixed routing policy differs")
    attempts = [a for row in rows for frame in row.get("frames", []) for a in frame.get("attempts", [])]
    summary = provider_summary(attempts, context)
    if not summary["complete"] or metadata.get("provider_observed") != summary:
        raise ValueError("actual provider/model evidence is incomplete or inconsistent")
    for row in rows:
        if "error" in row:
            continue
        frames = row.get("frames", [])
        if len(frames) != row.get("frames_planned"):
            raise ValueError("complete per-frame provider evidence required")
        scores = []
        for frame in frames:
            good = [a for a in frame.get("attempts", []) if a.get("ok")]
            if len(good) != 1 or good[0].get("score") != frame.get("score"):
                raise ValueError("frame score does not match its successful receipt")
            score = frame.get("score")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid per-frame score")
            scores.append(score)
        if not scores or max(scores) != row.get("score"):
            raise ValueError("clip score does not match full-precision frame evidence")


def split_groups(path, manifest_path, split):
    """Join frozen sidecar by question ID, filename AND media hash; no input edits."""
    data, digest = read_json(path)
    manifest, manifest_digest = read_json(manifest_path)
    if data.get("schema_version") != 1 or split not in {"validation", "test"}:
        raise ValueError("unsupported split manifest")
    if data.get("status") != "validation_membership_and_labels_frozen_with_recorded_limits":
        raise ValueError("source-group sidecar must preserve the frozen review scope")
    maps = {}
    for part in ("validation", "test"):
        section = data[part]
        mapping, seen_files, seen_hashes = {}, set(), {}
        for member in section["members"]:
            filename, media_hash, group = (member.get(k) for k in ("filename", "sha256", "source_video_group"))
            if (not isinstance(filename, str) or not filename or filename in seen_files
                    or not isinstance(media_hash, str) or len(media_hash) != 64
                    or any(c not in "0123456789abcdef" for c in media_hash)
                    or not isinstance(group, str) or not group.strip() or group != group.strip()):
                raise ValueError("invalid or duplicate source-group member")
            if media_hash in seen_hashes and seen_hashes[media_hash] != group:
                raise ValueError("same media hash has contradictory source groups")
            seen_files.add(filename)
            seen_hashes[media_hash] = group
            numbers = member.get("question_numbers")
            if not isinstance(numbers, list) or not numbers:
                raise ValueError("source-group member needs question numbers")
            for number in numbers:
                if type(number) is not int or number in mapping:
                    raise ValueError("duplicate or invalid source-group question number")
                mapping[number] = member
        if len(mapping) != section["questions"] or len(seen_files) != section["videos"]:
            raise ValueError("source-group membership counts differ")
        maps[part] = mapping
    for field in ("filename", "sha256", "source_video_group"):
        if ({r[field] for r in maps["validation"].values()}
                & {r[field] for r in maps["test"].values()}):
            raise ValueError("validation/test media overlap by " + field)
    section = data[split]
    if (section["manifest_sha256"] != manifest_digest
            or section["samples_json_sha256"] != manifest["samples_json_sha256"]):
        raise ValueError("split/manifest fingerprint differs")
    files = manifest["files"]
    if len(files) != len(maps[split]) or {r["number"] for r in files} != set(maps[split]):
        raise ValueError("split does not match exact manifest membership")
    for row in files:
        member = maps[split][row["number"]]
        if any(row[field] != member[field] for field in ("filename", "sha256")):
            raise ValueError("split question identity differs from manifest")
        if "source_video_group" in row and row["source_video_group"] != member["source_video_group"]:
            raise ValueError("inline and sidecar source groups disagree")
    return ({number: row["source_video_group"] for number, row in maps[split].items()},
            {"sha256": digest, "split": split, "status": data.get("status"),
             "train": data.get("train"), "development": data.get("development"),
             "limits": data.get("limits", []), "normalization": data.get("source_group_normalization"),
             "source_review_scope": "Existing recorded source mapping; label approval does not establish exhaustive source/scene isolation."})
