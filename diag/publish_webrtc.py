#!/usr/bin/env python3
"""Local synthetic WebRTC (VP8) publisher — a stand-in for the phone.

Reads a video source (a file, or an infinite lavfi test pattern) and publishes it
to MediaMTX over WHIP as VP8, so the harness can pull it via WHEP. Lets us run the
whole phone->harness path locally, with no phone.

    python diag/publish_webrtc.py --url https://192.168.1.228:8889/phonecam/whip \
        --cert diag/certs/server.crt
    python diag/publish_webrtc.py --url ... --source fight.mpeg
"""
from __future__ import annotations

import argparse
import asyncio
import ssl
import sys


async def publish(whip_url: str, cert: str | None, source: str, fmt: str | None, secs: float) -> int:
    import aiohttp
    from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription
    from aiortc.contrib.media import MediaPlayer

    player = MediaPlayer(source, format=fmt) if fmt else MediaPlayer(source)
    if player.video is None:
        print("[pub] !! source has no video track")
        return 2

    pc = RTCPeerConnection()
    pc.addTrack(player.video)

    # Force VP8 (aiortc decodes browser H264 poorly; VP8 is the reliable path).
    vp8 = [c for c in RTCRtpSender.getCapabilities("video").codecs if c.mimeType == "video/VP8"]
    for t in pc.getTransceivers():
        if t.sender and t.sender.track and t.sender.track.kind == "video":
            try:
                t.setCodecPreferences(vp8)
            except Exception as e:
                print(f"[pub] (could not pin VP8: {e})")

    ctx: ssl.SSLContext | bool = False
    if whip_url.startswith("https://"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # self-signed rig cert

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(whip_url, data=pc.localDescription.sdp,
                                 headers={"Content-Type": "application/sdp"}, ssl=ctx) as r:
                body = await r.text()
                print(f"[pub] WHIP POST -> HTTP {r.status}")
                if r.status not in (200, 201):
                    print(body[:300]); return 3
                await pc.setRemoteDescription(RTCSessionDescription(sdp=body, type="answer"))
    except Exception as e:
        print(f"[pub] !! WHIP failed: {type(e).__name__}: {e}")
        return 4

    print(f"[pub] publishing VP8 for {secs:.0f}s (Ctrl-C to stop)")
    try:
        await asyncio.sleep(secs)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    await pc.close()
    print("[pub] stopped")
    return 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="MediaMTX WHIP url, https://IP:8889/<path>/whip")
    ap.add_argument("--cert", help="(ignored; verify is disabled for the rig cert)")
    ap.add_argument("--source", default="testsrc=size=1280x720:rate=30",
                    help="video file, or an lavfi source string (default: test pattern)")
    ap.add_argument("--format", default=None, help="force input format, e.g. lavfi")
    ap.add_argument("--seconds", type=float, default=120.0)
    args = ap.parse_args(argv)
    fmt = args.format
    if fmt is None and ("=" in args.source and "/" not in args.source):
        fmt = "lavfi"  # a lavfi source string like testsrc=...
    return asyncio.run(publish(args.url, args.cert, args.source, fmt, args.seconds))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
