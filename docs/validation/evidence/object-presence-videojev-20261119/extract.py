"""Extract 1 fps 720-wide JPEGs for the 10 bag clips (local, no GPU)."""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tools.bag_demo.media import duration, frame_jpeg
from tools.bag_demo.server import TRUTH
media, out = Path(sys.argv[2]), Path(sys.argv[3])
index = {}
for clip in TRUTH:
    d = duration(media / clip)
    times = [float(t) for t in range(0, int(d) + 1) if t <= d - 0.05]
    frames = []
    for i, t in enumerate(times):
        p = out / f"{clip}.{i:03d}.jpg"
        p.write_bytes(frame_jpeg(media / clip, t, width=720))
        frames.append([i, t, p.name])
    index[clip] = {"duration": d, "frames": frames, "truth": TRUTH[clip]}
    print(clip, f"{d:.1f}s", len(frames), "frames", flush=True)
(out / "index.json").write_text(json.dumps(index, indent=1))
