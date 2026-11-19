"""Real-time acceptance run for the object presence node (PR #121).

Each clip is replayed at wall-clock speed from pre-extracted 2.5 fps frames: the
"current frame" is the newest frame whose timestamp <= elapsed time. The PR's
ObjectPresenceManager runs on its own timer thread (sample_fps 1, production
10 s client timeout). With --parallel, a second thread imitates the System 1
event branch: every 0.8 s (observation_fps 1.25) a 2-frame window 0.5 s apart with
the six built-in scene questions, on its own session, 20 s client timeout.
Writes one JSON line per object record and per event request.
"""
import argparse, base64, copy, json, sys, threading, time, uuid
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--endpoint", default="http://127.0.0.1:8097/predict")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--parallel", action="store_true")
a = ap.parse_args()
sys.path.insert(0, str(Path(a.repo) / "src"))
import httpx
from goodvision_harness import object_presence as op
from goodvision_harness.event_sensitivity import EVENT_SENSITIVITY_DEFAULTS
from goodvision_harness.videojev_contract import scene_question_template

frames_dir = Path(a.frames)
index = json.loads((frames_dir / "index.json").read_text())
objects = [copy.deepcopy(op.PRESETS["knife"]), copy.deepcopy(op.PRESETS["alcohol_bottle"])]
event_template = [dict(r, query_parts=list(r["query_parts"]))
                  for r in scene_question_template(labels=tuple(EVENT_SENSITIVITY_DEFAULTS))]
lock = threading.Lock()
out = open(a.out, "a")


def emit(row):
    with lock:
        out.write(json.dumps(row) + "\n"); out.flush()


for clip, item in index.items():
    frames = [(i, t, (frames_dir / name).read_bytes()) for i, t, name in item["frames"]]
    duration = item["duration"]
    start = time.monotonic()

    def current(offset=0.0):
        now = time.monotonic() - start - offset
        if now < 0:
            return None
        best = None
        for f in frames:
            if f[1] <= now:
                best = f
        return best

    session = f"{a.tag}-{clip}-{uuid.uuid4().hex[:6]}"
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": objects, "sample_fps": 1, "clear_after": 3},
        frame_provider=current, post=op.http_post(a.endpoint, timeout_seconds=10.0),
        stream_session_id=session,
        on_result=lambda r, clip=clip, item=item: emit({
            "kind": "object", "tag": a.tag, "clip": clip, "truth": item["truth"],
            "wall": time.monotonic() - start, **r}),
    )
    stop = threading.Event()

    def events():
        client = httpx.Client(timeout=20.0, trust_env=False)
        sent_template = False
        next_due = time.monotonic()
        while not stop.is_set():
            newest, older = current(), current(0.5)
            if newest and older and newest[1] > older[1]:
                body = {"request_id": uuid.uuid4().hex, "session": session + ":events",
                        "fps": 2.0, "frame_times": [older[1], newest[1]],
                        "images": [base64.b64encode(f[2]).decode() for f in (older, newest)]}
                if not sent_template:
                    body["question_template"] = event_template
                t0 = time.monotonic(); row = {"kind": "event", "tag": a.tag, "clip": clip,
                                               "wall": t0 - start}
                try:
                    r = client.post(a.endpoint, json=body)
                    v = r.json() if r.content else {}
                    row.update(status=r.status_code,
                               error=None if r.status_code == 200 else (v.get("error") if isinstance(v, dict) else "http"))
                    if r.status_code == 200:
                        sent_template = True
                        row["answers"] = {x["id"]: [round(p, 3) for p in x["probabilities"]]
                                          for x in v.get("raw", []) if isinstance(x, dict)}
                except Exception as e:
                    row.update(status=None, error=type(e).__name__)
                row["seconds"] = time.monotonic() - t0
                emit(row)
            next_due += 0.8
            stop.wait(max(0.0, next_due - time.monotonic()))
        client.close()

    worker = threading.Thread(target=events, daemon=True) if a.parallel else None
    manager.start()
    if worker:
        worker.start()
    time.sleep(duration + 0.5)
    stop.set(); manager.stop(timeout=15)
    if worker:
        worker.join(25)
    st = manager.status()
    print(a.tag, clip, "objects ok", st["succeeded"], "failed", st["failed"], "last_error", st["last_error"], flush=True)
