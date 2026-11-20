"""Drive the PR #121 ObjectPresenceManager over pre-extracted bag clip frames
against a real VideoJev service (tunneled). One fresh manager/session per clip.
Writes every tick (probabilities, decisions, latch, transport seconds)."""
import argparse, copy, json, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--endpoint", default="http://127.0.0.1:8097/predict")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.repo) / "src"))
from goodvision_harness import object_presence as op

frames = Path(a.frames)
index = json.loads((frames / "index.json").read_text())
objects = [copy.deepcopy(op.PRESETS["knife"]), copy.deepcopy(op.PRESETS["alcohol_bottle"])]
post = op.http_post(a.endpoint, timeout_seconds=60.0)
with open(a.out, "a") as out:
    for clip, item in index.items():
        seq = iter([(i, t, (frames / name).read_bytes()) for i, t, name in item["frames"]])
        manager = op.ObjectPresenceManager(
            {"id": "objects", "objects": objects, "clear_after": 3},
            frame_provider=lambda: next(seq, None),
            post=post, stream_session_id=f"{a.tag}-{clip}",
        )
        for _ in item["frames"]:
            record = manager.tick()
            if record is None:
                continue
            out.write(json.dumps({"tag": a.tag, "clip": clip, "truth": item["truth"], **record}) + "\n")
            out.flush()
            objs = " ".join(f"{o['id']}={o['present_probability'] if o['present_probability'] is None else round(o['present_probability'],3)}{'*' if o['present'] else ''}" for o in record["objects"])
            print(a.tag, clip, f"t={record['timestamp_seconds']:.1f}", record["error"] or "", objs, f"{record['transport_seconds']:.2f}s", flush=True)
