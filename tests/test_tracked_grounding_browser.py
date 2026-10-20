"""Chromium regression: twelve videos with independent ID/box/hidden displays."""

import os
import threading
import time
from fractions import Fraction
from types import SimpleNamespace

import pytest
from PIL import Image
from test_vlm_monitor import job, result

from tools.tracked_grounding.monitor import PersonMonitor
from tools.tracked_grounding.suite import arguments
from tools.vlm_multistream_benchmark import decode


@pytest.mark.skipif(
    os.environ.get("VLM_MONITOR_BROWSER_TESTS") != "1",
    reason="opt-in installed Chromium regression",
)
@pytest.mark.parametrize(
    "profile,display",
    [("cloud", "ids"), ("decisions", "ids"), ("fp8", "ids"),
     ("fp8", "boxes"), ("fp8", "off")],
)
@pytest.mark.parametrize("mode", ["events", "compact", "parallel"])
def test_ids_only_and_old_positions_hidden_in_all_modes(
    tmp_path, monkeypatch, mode, profile, display
):
    from playwright.sync_api import sync_playwright

    av = pytest.importorskip("av")
    source = tmp_path / "source.mp4"
    dimensions = {"events": (96, 64), "compact": (192, 64), "parallel": (64, 96)}
    width, height = dimensions[mode]
    with av.open(str(source), "w") as writer:
        stream = writer.add_stream("libx264", rate=15)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for n in range(60):
            f = av.VideoFrame.from_image(
                Image.new("RGB", (width, height), (n * 3, 70, 40))
            )
            f.pts, f.time_base = n, Fraction(1, 15)
            for packet in stream.encode(f):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    if profile == "decisions":
        from tools.tracked_decisions.suite import arguments as decision_arguments

        args = decision_arguments()
    elif profile == "fp8":
        from tools.vast_vlm.fp8_adapter import arguments as fp8_arguments

        args = fp8_arguments()
    else:
        args = arguments()
    # Stray #48 preview settings must not create extra detector/model calls.
    monkeypatch.setenv("PREVIEW_DETECTOR_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("PREVIEW_SELECT_EVENTS", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-preview-key")
    args.person_display = display
    args.record = profile in {"decisions", "fp8"} and display == "ids"
    frames = decode(source, 2.5, 720, 90)
    requests = []
    broker = SimpleNamespace(
        condition=threading.Condition(),
        latest={
            i: {
                "state": "ready",
                "frame_sequence": 0,
                "timestamp_seconds": 0,
                "people": [
                    {
                        "track_id": "P0001",
                        "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9},
                    }
                ],
            }
            for i in range(12)
        },
    )
    with PersonMonitor(tmp_path, [frames], args) as monitor:
        monitor.broker = broker
        stage = tmp_path / "stage"
        stage.mkdir()
        monitor.begin({**job(), "streams": 12, "mode": mode, "duration": 3}, stage)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.add_init_script("""window.drawnIds=[];window.boxes=0;
              const p=CanvasRenderingContext2D.prototype;
              const text=p.fillText, rect=p.strokeRect;
              p.fillText=function(t,...a){
                this.canvas.lastLabel={text:t,x:a[0],y:a[1]};
                if(window.drawnIds.length<100)window.drawnIds.push(t);
                return text.call(this,t,...a)};
              p.strokeRect=function(...a){window.boxes++;return rect.apply(this,a)};
            """)
            page.on("request", lambda request: requests.append(request.url))
            page.goto(monitor.url + "?tracking=on")
            page.wait_for_function("window.monitorReady")
            assert page.locator("video").count() == 12
            assert page.locator("canvas").count() == (0 if display == "off" else 12)
            assert page.locator(".framepair .screen").count() == 12
            monitor.source_started(time.monotonic(), [0] * 12)
            if display != "off":
                page.wait_for_function("window.drawnIds.includes('P0001')")
                assert page.locator("canvas").first.evaluate("""c=>{
                  const v=c.parentElement.querySelector('video');
                  const w=v.videoWidth,h=v.videoHeight,k=Math.min(c.width/w,c.height/h);
                  const x=(c.width-w*k)/2+.3*w*k;
                  return c.lastLabel.text==='P0001' && Math.abs(c.lastLabel.x-x)<1;
                }""")
            page.wait_for_function("document.querySelector('video').currentTime>.9")
            # No refreshed detector position: text must disappear, not move with
            # an unrelated current person or persist on the last known position.
            if display != "off":
                assert page.locator("canvas").first.evaluate("""c=>{
                  const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
                  return !d.some((v,i)=>i%4===3&&v>0);
                }""")
            assert (page.evaluate("window.boxes") > 0) == (display == "boxes")
            if mode != "events":
                r = result(3, "grounding")
                r.update(
                    subject_id_scope="stream_session",
                    observations=[
                        {
                            "event_label": "smoking",
                            "subject_ids": ["det-camera-P0001"],
                            "detected": True,
                        }
                    ],
                )
                r["objects"][0].update(
                    object_id="det-camera-P0001", attributes={"track_id": "P0001"}
                )
                monitor.result(0, r)
                old = {**r, "message_id": "old", "historical_only": True}
                old["observations"] = [{**r["observations"][0], "detected": False}]
                monitor.result(0, old)
                page.wait_for_function(
                    "document.querySelector('.evidence-note').textContent.includes('吸烟有')"
                )
                assert page.locator(".evidence-note").first.is_visible()
            page.wait_for_function(
                "window.monitorDiagnostics.playback.every(r=>r.seconds>=2)"
            )
            monitor.draining()
            recording = monitor.finish(
                {
                    "lanes": {},
                    "input_timing_valid": True,
                    "all_drained": True,
                    "result_records_missing": 0,
                }
            )
            if args.record:
                assert recording["complete"], recording
                assert (stage / "recording/viewport.mp4").is_file()
            page.wait_for_function(
                "window.monitorDiagnostics.phases.includes('finished')"
            )
            page.wait_for_timeout(400)
            page.screenshot(path=str(tmp_path / "monitor.png"))
            assert page.evaluate("window.finalPainted"), page.locator(
                ".camera, #summary"
            ).evaluate_all(
                "es=>es.map(e=>({bottom:e.getBoundingClientRect().bottom,height:innerHeight}))"
            )
            assert not any("/tracking/" in url or "/frame/" in url for url in requests)
            assert page.evaluate("window.monitorDiagnostics.allCamerasVisible")
            browser.close()
