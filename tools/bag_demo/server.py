"""Tiny FastAPI app for the 翻包 (bag-search) demo.

Left pane: the clip playing with YOLO boxes. Right pane: a VideoJev knife/weapon
candidate verdict as the clip plays, each window scored over the SSH tunnel
to the rented GPU's native scorer (:8093). Demo-only; separate from the frozen benchmark
and from the production console.

Run:  VIDEOJEV_ENDPOINT=http://127.0.0.1:8093 \
      PYTHONPATH=src .venv/bin/python -m tools.bag_demo.server
"""

import json
import math
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tools.bag_demo import videojev_bag as vb
from tools.bag_demo.media import duration

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
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
SCORING = threading.BoundedSemaphore(1)


@app.middleware("http")
async def local_browser_requests(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.headers.get("sec-fetch-site") == "cross-site" or (
        origin and origin != f"{request.url.scheme}://{request.headers.get('host')}"
    ):
        return JSONResponse({"error": "cross_origin_request"}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


def _local_file(name):
    root = MEDIA_DIR.resolve()
    path = root / name
    if Path(name).name != name or path.is_symlink():
        raise HTTPException(404, "unknown clip")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(404, "unknown clip") from None
    if resolved.parent != root or not resolved.is_file():
        raise HTTPException(404, "unknown clip")
    return resolved


def _clips():
    if not MEDIA_DIR.exists():
        return []
    present = {
        p.name
        for p in MEDIA_DIR.iterdir()
        if p.suffix.lower() == ".mov" and p.is_file() and not p.is_symlink()
    }
    ordered = [c for c in CLIP_ORDER if c in present]
    ordered += sorted(present - set(ordered))
    return ordered


def _safe_clip(name):
    if name not in _clips():
        raise HTTPException(404, "unknown clip")
    return _local_file(name)


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.read_text(encoding="utf-8")


@app.get("/api/clips")
def clips():
    return {
        "clips": [{"name": c, "truth": TRUTH.get(c)} for c in _clips()],
        "alcohol_status": "not_integrated",
        "demo_only": True,
    }


@app.get("/media/{name}")
def media(name: str):
    return FileResponse(_safe_clip(name))


@app.get("/api/boxes/{name}")
def boxes(name: str):
    _safe_clip(name)
    box_name = name + ".boxes.json"
    if not (MEDIA_DIR / box_name).exists():
        return JSONResponse(
            {"frames": [], "note": "no precomputed boxes"}, status_code=200
        )
    box_file = _local_file(box_name)
    try:
        if box_file.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("boxes too large")
        data = json.loads(box_file.read_text(encoding="utf-8"))
        _validate_boxes(data, name)
        return JSONResponse(data)
    except (ValueError, TypeError, KeyError, OSError):
        return JSONResponse({"frames": [], "error": "invalid_boxes"}, status_code=422)


def _validate_boxes(data, name):
    if not isinstance(data, dict) or data.get("clip") != name:
        raise ValueError("boxes do not belong to this clip")
    if not isinstance(data.get("frames"), list):
        raise ValueError("invalid frames")
    previous = -1
    for frame in data["frames"]:
        if not isinstance(frame, dict):
            raise ValueError("invalid frame")
        t = frame.get("t")
        if type(t) not in (int, float) or not math.isfinite(t) or t < 0 or t < previous:
            raise ValueError("invalid box timestamp")
        previous = t
        if not isinstance(frame.get("boxes"), list):
            raise ValueError("invalid boxes")
        for box in frame["boxes"]:
            if not isinstance(box, dict):
                raise ValueError("invalid box")
            xyxy = box.get("xyxy")
            conf = box.get("conf")
            if (
                not isinstance(xyxy, list)
                or len(xyxy) != 4
                or any(
                    type(v) not in (int, float)
                    or not math.isfinite(v)
                    or not 0 <= v <= 1
                    for v in xyxy
                )
                or xyxy[0] > xyxy[2]
                or xyxy[1] > xyxy[3]
                or type(conf) not in (int, float)
                or not math.isfinite(conf)
                or not 0 <= conf <= 1
                or not isinstance(box.get("cls"), str)
                or type(box.get("alert")) is not bool
            ):
                raise ValueError("invalid box")
    json.dumps(data, allow_nan=False)


@app.get("/api/verdict")
def verdict(clip: str, t: float = Query(..., ge=0.0, allow_inf_nan=False)):
    path = _safe_clip(clip)
    if not SCORING.acquire(blocking=False):
        return JSONResponse({"error": "scorer_busy"}, status_code=503)
    step = 1.0 / vb.FPS
    try:
        if t >= duration(path):
            raise HTTPException(422, "timestamp outside clip")
        # Require two real past frames; never score unseen future frames at startup.
        if t < step:
            return {
                "clip": clip,
                "t": t,
                "status": "insufficient_frames",
                "knife": None,
                "weapon_native": None,
                "knife_present": False,
                "knife_p": None,
                "alcohol_p": None,
                "alcohol": None,
                "label": "存疑",
                "alcohol_status": "not_integrated",
            }
        times = [t - step * k for k in reversed(range(4)) if t - step * k >= 0]
        images = vb.sample_frames(path, times)
        result = vb.predict(
            images,
            times,
            endpoint=ENDPOINT,
            request_id=f"{clip}:{t:.1f}",
            session=clip,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Exception strings may contain endpoint credentials, filenames or replies.
        return JSONResponse(
            {"error": type(exc).__name__, "status": "unavailable"}, status_code=502
        )
    finally:
        SCORING.release()
    result["clip"] = clip
    result["t"] = t
    result["truth"] = TRUTH.get(clip)
    result["status"] = "ok"
    result["alcohol_status"] = "not_integrated"
    return result


def main():
    import uvicorn

    port = int(os.environ.get("BAG_DEMO_PORT", "8099"))
    host = os.environ.get("BAG_DEMO_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("bag-demo is unauthenticated and must bind to loopback")
    print(
        f"bag-demo on http://{host}:{port} (demo only; hosted alcohol not integrated)"
    )
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
