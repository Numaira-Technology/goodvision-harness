"""Real GPU service checks using explicit JPEGs and the shipped workflow clients.

Run in the Main image on an isolated test host. This tool calls already-running
services; renting, installing, restarting and cleanup belong to the operator.
"""

import argparse
import base64
import json
import time
import urllib.request
import uuid
from pathlib import Path


def http(method, url, body=None, headers=None):
    data = (
        body
        if isinstance(body, bytes)
        else json.dumps(body).encode()
        if body is not None
        else None
    )
    hdr = {"Content-Type": "application/json", **(headers or {})}
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=hdr, method=method), timeout=180
    ) as r:
        return json.load(r)


def events(args, frames):
    from goodvision_harness.model_api import ModelAPISettings
    from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
    from goodvision_harness.scene_runtime import TemporalInspection
    from goodvision_harness.vlm_independent import (
        IndependentEventState,
        SourceSample,
        VLMProcessConfig,
        _inspector,
        utc_now,
    )

    model = (
        "ZhuYuhan/VideoJev-4B"
        if args.module == "system1"
        else "openai/clip-vit-large-patch14"
    )
    port = 28097 if args.module == "system1" else 28098
    base = args.url or f"http://127.0.0.1:{port}"
    health = http("GET", base + "/health")
    assert health["ready"] is True
    reports = []
    for lane in range(2):
        session = f"{args.module}-{args.suffix}-{lane}-{uuid.uuid4().hex[:8]}"
        cfg = VLMProcessConfig(
            session,
            "public-video-" + str(lane),
            1,
            ("smoking", "fall"),
            ModelAPISettings(
                provider="local",
                endpoint=base.rstrip("/") + "/predict",
                timeout_seconds=150,
            ),
            model,
            observation_fps=2,
            observation_frame_delta_seconds=0.5,
            output_mode="events",
        )
        inspector = _inspector(cfg)
        state = IndependentEventState(cfg)
        try:
            for batch in range(2):
                evidence = tuple(
                    EvidenceFrame(
                        session, batch * 4 + i, batch * 2 + i * 0.5, f.read_bytes()
                    )
                    for i, f in enumerate(frames[batch * 4 : batch * 4 + 4])
                )
                request = InspectionRequest(
                    f"{session}-{batch}",
                    (),
                    "independent_source",
                    cfg.labels,
                    evidence[0].timestamp_seconds,
                    evidence[-1].timestamp_seconds,
                )
                job = TemporalInspection(session, cfg.source_id, request, evidence)
                started = time.monotonic()
                verdicts = inspector.inspect(job)
                report = state.report(
                    job,
                    SourceSample(evidence[-1], utc_now(), time.monotonic()),
                    verdicts,
                    {"model_seconds": time.monotonic() - started},
                )
                assert (
                    report["analysis_complete"]
                    and report["state"] == "completed"
                    and len(report["judgments"]) == 2
                ), report
                assert (
                    report["stream_session_id"] == session
                    and report["request_id"] == request.request_id
                )
                reports.append(report)
        finally:
            inspector.close()
    return {
        "health": health,
        "workflow_reports": reports,
        "distinct_session_count": 2,
        "real_model_client_and_workflow_report": True,
    }


def bbox(args, frames):
    base = args.url or "http://127.0.0.1:28096"
    health = http("GET", base + "/health/ready")
    assert health["ready"] and health["device"] == "cuda", health
    responses = []
    for lane in range(2):
        session = f"bbox-{args.suffix}-{lane}-{uuid.uuid4().hex[:8]}"
        for i, f in enumerate(frames[:4]):
            r = http(
                "POST",
                base + "/v1/tracks",
                f.read_bytes(),
                {
                    "Content-Type": "image/jpeg",
                    "x-goodvision-stream-session-id": session,
                    "x-goodvision-frame-sequence": str(i),
                    "x-goodvision-timestamp-seconds": str(i * 0.5),
                },
            )
            assert r["stream_session_id"] == session and r["frame_sequence"] == i
            persons = [
                x
                for x in r["objects"]
                if x["label"] == "person" and x.get("observation_state") == "observed"
            ]
            assert persons, r
            for x in persons:
                assert (
                    0 <= x["bbox"]["x1"] < x["bbox"]["x2"] <= 1
                    and 0 <= x["bbox"]["y1"] < x["bbox"]["y2"] <= 1
                )
            responses.append(r)
    return {
        "health": health,
        "responses": responses,
        "nonempty_person_boxes": True,
        "independent_sessions": True,
    }


def contour(args, frames):
    base = args.url or "http://127.0.0.1:28093"
    key = args.contour_key_file.read_text().strip()
    headers = {"Authorization": "Bearer " + key}
    health = http("GET", base + "/health/ready", headers=headers)
    responses = []
    sessions = []
    for lane in range(2):
        sid = f"contour-{args.suffix}-{lane}-{uuid.uuid4().hex[:8]}"
        opened = http(
            "POST", base + "/v1/contour/sessions", {"stream_session_id": sid}, headers
        )
        sessions.append(opened["video_session_id"])
        url = base + "/v1/contour/sessions/" + opened["video_session_id"]
        try:
            for i, f in enumerate(frames[:4]):
                body = {
                    "stream_session_id": sid,
                    "request_id": f"{sid}-{i}",
                    "is_last_batch": i == 3,
                    "frames": [
                        {
                            "frame_sequence": i,
                            "timestamp_seconds": i * 0.5,
                            "jpeg_base64": base64.b64encode(f.read_bytes()).decode(),
                        }
                    ],
                }
                r = http("POST", url + "/append", body, headers)
                assert (
                    r["stream_session_id"] == sid
                    and r["request_id"] == body["request_id"]
                )
                assert (
                    r["processed_frame_sequence"] == i and r["pending_frame_count"] == 0
                )
                responses.append(r)
        finally:
            http("DELETE", url, headers=headers)
    assert len(set(sessions)) == 2
    objects = [
        obj
        for response in responses
        for frame in response["frames"]
        for obj in frame["objects"]
    ]
    assert objects, "Contour emitted no confirmed objects"
    assert any(obj.get("attributes", {}).get("mask_contours") for obj in objects), (
        "Contour emitted no mask geometry"
    )
    return {
        "health": health,
        "responses": responses,
        "distinct_sessions_closed": sessions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=("bbox", "contour", "system1", "neural"))
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contour-key-file", type=Path)
    parser.add_argument("--url")
    parser.add_argument("--suffix", default="initial")
    args = parser.parse_args()
    frames = sorted(args.frames_dir.glob("*.jpg"))
    if len(frames) < 8:
        parser.error(
            "provide at least eight ordered JPEG frames from a public test clip"
        )
    if args.module == "contour" and args.contour_key_file is None:
        parser.error("Contour requires --contour-key-file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = {"module": args.module, "suffix": args.suffix, "started_at": started}
    try:
        check = (
            events
            if args.module in ("system1", "neural")
            else bbox
            if args.module == "bbox"
            else contour
        )
        result.update(check(args, frames))
        result["passed"] = True
    except Exception as error:
        result.update(passed=False, error=type(error).__name__)
        for name in ("code", "status_code"):
            value = getattr(error, name, None)
            if isinstance(value, (str, int)):
                result[name] = value
        raise SystemExit(1) from None
    finally:
        result["elapsed_seconds"] = round(time.time() - started, 3)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in ("module", "passed", "elapsed_seconds", "error")
                }
            )
        )


if __name__ == "__main__":
    main()
