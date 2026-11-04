"""Tiny FastAPI app for the 翻包 (bag-search) demo.

Left pane: the clip playing with YOLO boxes. Right pane: the live VideoJev verdict
(武器 / 酒瓶 / 无违禁品 / 存疑) as the clip plays, each window scored over the SSH tunnel
to the rented GPU's native scorer (:8093). Demo-only; separate from the frozen benchmark
and from the production console.

Run:  VIDEOJEV_ENDPOINT=http://127.0.0.1:8093 \
      PYTHONPATH=src .venv/bin/python -m tools.bag_demo.server
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from tools.bag_demo import videojev_bag as vb

REPO_ROOT = Path(__file__).resolve().parents[2]
MEDIA_DIR = Path(os.environ.get("BAG_DEMO_MEDIA", REPO_ROOT / "demo-media" / "bag"))
PAGE = Path(__file__).with_name("page.html")
ENDPOINT = os.environ.get("VIDEOJEV_ENDPOINT", "http://127.0.0.1:8093")

# Ground truth (manifest_120 #101-120) — used only for an optional self-check badge.
TRUTH = {
    "IMG_2729.MOV": {"alcohol": True, "knife": False},
    "IMG_2730.MOV": {"alcohol": False, "knife": False},
    "IMG_2731.MOV": {"alcohol": False, "knife": True},
    "IMG_2732.MOV": {"alcohol": False, "knife": True},
    "IMG_2733.MOV": {"alcohol": False, "knife": False},
    "IMG_1185.MOV": {"alcohol": False, "knife": False},
    "IMG_1186.MOV": {"alcohol": False, "knife": False},
    "IMG_1187.MOV": {"alcohol": False, "knife": False},
    "IMG_1188.MOV": {"alcohol": False, "knife": False},
    "IMG_1189.MOV": {"alcohol": False, "knife": False},
}
CLIP_ORDER = list(TRUTH)

app = FastAPI(title="bag-demo")


def _clips():
    if not MEDIA_DIR.exists():
        return []
    present = {p.name for p in MEDIA_DIR.glob("*.MOV")} | {p.name for p in MEDIA_DIR.glob("*.mov")}
    ordered = [c for c in CLIP_ORDER if c in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _safe_clip(name):
    if name not in _clips():
        raise HTTPException(404, "unknown clip")
    return MEDIA_DIR / name


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.read_text(encoding="utf-8")


@app.get("/api/clips")
def clips():
    return {"clips": [{"name": c, "truth": TRUTH.get(c)} for c in _clips()],
            "endpoint": ENDPOINT}


@app.get("/media/{name}")
def media(name: str):
    return FileResponse(_safe_clip(name))


@app.get("/api/boxes/{name}")
def boxes(name: str):
    _safe_clip(name)
    box_file = MEDIA_DIR / (name + ".boxes.json")
    if not box_file.exists():
        return JSONResponse({"frames": [], "note": "no precomputed boxes"}, status_code=200)
    return JSONResponse(json.loads(box_file.read_text()))


@app.get("/api/verdict")
def verdict(clip: str, t: float = Query(..., ge=0.0)):
    path = _safe_clip(clip)
    # Window of up to 4 frames ending at t, spaced 1/fps, clamped at the clip start.
    step = 1.0 / vb.FPS
    times = sorted({round(max(0.0, t - step * k), 3) for k in range(4)})
    if len(times) < 2:
        times = [0.0, round(step, 3)]
    try:
        images = vb.sample_frames(path, times)
        result = vb.predict(
            images, times, endpoint=ENDPOINT,
            request_id=f"{clip}:{t:.1f}", session=clip,
        )
    except Exception as exc:  # surface scorer/tunnel errors to the page, don't 500 silently
        return JSONResponse({"error": type(exc).__name__, "detail": str(exc)[:300]},
                            status_code=502)
    result["clip"] = clip
    result["t"] = t
    result["truth"] = TRUTH.get(clip)
    return result


def main():
    import uvicorn

    port = int(os.environ.get("BAG_DEMO_PORT", "8099"))
    host = os.environ.get("BAG_DEMO_HOST", "127.0.0.1")
    print(f"bag-demo on http://{host}:{port}  (VideoJev → {ENDPOINT}, media → {MEDIA_DIR})")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
