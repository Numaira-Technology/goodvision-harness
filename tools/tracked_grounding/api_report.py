"""Offline cloud-VLM API comparison, with all planned windows as denominator."""

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tools.vlm_benchmark.accounting import complete
from tools.vlm_benchmark.model_comparison import safe_path, sha
from tools.vlm_benchmark.report import write_json
from tools.vlm_multistream_benchmark import planned_observations, report_lane, timing


def api_seconds(report, *, transport="nonstream"):
    """Nonstream HTTP send through full body; exclude build/parse/local work."""
    h = report.get("timing", {}).get("http_timing", {})
    keys = (
        "http_until_headers_seconds",
        "response_read_seconds",
        "request_build_seconds",
    )
    if not all(
        type(h.get(k)) in (int, float) and math.isfinite(h[k]) and h[k] >= 0
        for k in keys
    ):
        return None
    if transport not in {"nonstream", "sse"}:
        raise ValueError("unknown API timing transport")
    # SSE transport starts its header timer AFTER packing, pacing and build.
    value = h[keys[0]] + h[keys[1]] - (h[keys[2]] if transport == "nonstream" else 0)
    return value if value >= 0 else None


def summarize(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    model = summary.get("model")
    transport = summary.get("api_transport", "nonstream")
    provider = summary.get("provider", "")
    # Historical streaming tests predate the tracked suite's summary envelope.
    # Read their frozen manifest; never rewrite original evidence to fit a schema.
    if model is None and summary.get("settings", {}).get("vision_profile") in {
        "qwen27_paced",
        "glm_low",
    }:
        from tools.vlm_benchmark.vision_profiles import PROFILES

        manifest = json.loads((root / "manifest.json").read_text())
        arm = summary["settings"]["vision_profile"]
        model = PROFILES[arm]["model"]
        if manifest.get("model") != model or summary["settings"].get("model") != model:
            raise ValueError("historical model identity differs from frozen profile")
        if summary["settings"].get("backend") != "openrouter":
            raise ValueError("historical candidate was not an OpenRouter run")
        provider, transport = "OpenRouter; historical " + arm, "sse"
    allowed = {
        ("qwen/qwen3.5-35b-a3b", "nonstream"),
        ("qwen/qwen3.8-27b", "sse"),
        ("z-ai/glm-5.3-flash", "sse"),
    }
    if (model, transport) not in allowed or not provider.startswith("OpenRouter"):
        raise ValueError("unrecognized cloud model API timing contract")
    jobs = [j for j in summary["jobs"] if j["phase"] == "measurement"]
    groups, pairs, denominators = defaultdict(list), defaultdict(dict), Counter()
    audit = []
    identities = set()
    for job in jobs:
        identity = (job["mode"], job["repeat"])
        if identity in identities:
            raise ValueError("duplicate measurement trial")
        identities.add(identity)
        result = job["result"]
        expected = planned_observations(
            result["duration_seconds"],
            result["observation_fps"],
            result.get("source_sample_fps"),
        )
        cameras = result["streams"]
        lanes = set(result["lanes"])
        for lane in lanes:
            denominators[job["mode"], lane] += expected * cameras
        if job["mode"] == "parallel":
            denominators["parallel", "joint"] += expected * cameras
        path = safe_path(root, job["raw_reports"])
        audit.append({"path": str(path), "sha256": sha(path)})
        seen, counts = set(), Counter()
        for line in path.read_text().splitlines():
            item = json.loads(line)
            r, camera = item["report"], item["camera"]
            lane, wid = report_lane(r), r.get("window_id")
            key = (camera, wid, lane)
            if (
                type(camera) is not int
                or not 1 <= camera <= cameras
                or lane not in lanes
                or not isinstance(wid, str)
                or not wid
                or key in seen
            ):
                raise ValueError("invalid or duplicate source/window/lane")
            seen.add(key)
            counts[camera, lane] += 1
            if counts[camera, lane] > expected:
                raise ValueError("recorded windows exceed planned cadence")
            if complete(r):
                value = api_seconds(r, transport=transport)
                groups[job["mode"], lane].append(value)
                if job["mode"] == "parallel":
                    pairs[job["repeat"], camera, wid][lane] = value
    rows = []
    for (mode, lane), planned in denominators.items():
        if lane == "joint":
            good = [p for p in pairs.values() if set(p) == {"events", "grounding"}]
            values = [
                max(p.values()) if all(v is not None for v in p.values()) else None
                for p in good
            ]
        else:
            values = groups[mode, lane]
        known = [v for v in values if v is not None]
        rows.append(
            {
                "mode": mode,
                "lane": lane,
                "planned": planned,
                "complete": len(values),
                "complete_rate": len(values) / planned,
                "api_timing_missing": len(values) - len(known),
                "api_seconds": timing(known) if lane != "joint" else None,
                "api_le3_rate": (
                    sum(v <= 3 for v in known) / planned
                    if len(known) == len(values)
                    else None
                ),
            }
        )
    expected_trials = {
        (m, r) for m in ("events", "compact", "parallel") for r in (1, 2)
    }
    return {
        "root": str(root),
        "model": model,
        "provider": provider,
        "api_transport": transport,
        "state": summary["state"],
        "rows": rows,
        "all_six_trials_finished": summary["state"] == "completed"
        and identities == expected_trials,
        "source_audit": audit,
        "scope": (
            f"{transport} API send through full response body; completed requests only"
        ),
        "parallel": (
            "same camera/window; both API durations <=3s; not joint wall-clock latency"
        ),
    }


def analyze(root, baseline=None):
    root = Path(root)
    current = summarize(root)
    previous = summarize(baseline) if baseline else None
    result = {
        "current": current,
        "baseline": previous,
        "comparison_note": (
            "Historical comparison, not simultaneous A/B; "
            "runtime and provider availability may differ."
        ),
    }
    write_json(root / "api-comparison.json", result)
    rows = []
    for name, run in (("current", current), ("historical", previous)):
        for row in (run or {}).get("rows", []):
            seconds = row["api_seconds"] or {}
            values = [
                name,
                row["mode"],
                row["lane"],
                f"{row['complete']}/{row['planned']}",
                f"{row['complete_rate']:.2%}",
                "unavailable"
                if row["api_le3_rate"] is None
                else f"{row['api_le3_rate']:.2%}",
                seconds.get("p50"),
                seconds.get("p95"),
            ]
            rows.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(v))}</td>" for v in values)
                + "</tr>"
            )
    (root / "api-comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Qwen API comparison</title>'
        "<style>body{font:16px system-ui;margin:32px}"
        "td,th{padding:8px;border:1px solid #ccc}"
        "table{border-collapse:collapse}</style>"
        "<h1>Cloud VLM + YOLO: API latency</h1>"
        f"<p>State: {html.escape(current['state'])}</p>"
        f"<p>{html.escape(current['scope'])}</p>"
        "<p>All planned windows remain in completion-rate denominators. "
        "Warmups and preflight excluded.</p>"
        f"<p>{html.escape(current['parallel'])}</p><p>{html.escape(result['comparison_note'])}</p>"
        "<table><tr><th>Run</th><th>Mode</th><th>Lane</th><th>Complete/planned</th>"
        "<th>Complete</th><th>API ≤3s</th><th>API P50(s)</th><th>API P95(s)</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", type=Path)
    options = parser.parse_args()
    analyze(options.root, options.baseline)
