#!/usr/bin/env python3
"""Standalone WHEP connectivity probe — pinpoints why the harness gets no video.

Mirrors the harness's aiortc WHEP flow but prints every stage: SSL, the WHEP POST
status, the SDP answer, ICE/connection state changes, negotiated codecs, and
whether a video frame actually arrives. Run it while the phone is publishing.

Usage:
    diag/../.venv/bin/python diag/whep_probe.py \
        "webrtc+https://192.168.1.225:8889/phonecam/whep" \
        --cert diag/certs/server.crt
"""

from __future__ import annotations

import argparse
import asyncio
import ssl
import sys
import time


async def probe(url: str, cert: str | None, wait_s: float, insecure: bool = False) -> int:
    import aiohttp
    from aiortc import RTCPeerConnection, RTCSessionDescription

    transport_url = url.replace("webrtc+https://", "https://").replace("webrtc+http://", "http://")
    print(f"[probe] WHEP endpoint: {transport_url}")

    # SSL context for the aiohttp POST: trust the cert, or disable verify.
    ssl_ctx: ssl.SSLContext | bool
    if not transport_url.startswith("https://"):
        ssl_ctx = False
    elif insecure:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        print("[probe] TLS verify OFF (--insecure)")
    else:
        ssl_ctx = ssl.create_default_context()
        if cert:
            try:
                ssl_ctx.load_verify_locations(cert)
                print(f"[probe] trusting cert: {cert}")
            except Exception as e:
                print(f"[probe] !! could not load cert {cert}: {e}")
        print("[probe] TLS verify ON")

    pc = RTCPeerConnection()
    track_ready = asyncio.get_event_loop().create_future()
    first_frame = asyncio.get_event_loop().create_future()

    @pc.on("iceconnectionstatechange")
    async def _ice():
        print(f"[probe] ICE state -> {pc.iceConnectionState}")

    @pc.on("connectionstatechange")
    async def _conn():
        print(f"[probe] connection state -> {pc.connectionState}")

    @pc.on("track")
    def _track(track):
        print(f"[probe] TRACK received: kind={track.kind}")
        if track.kind == "video" and not track_ready.done():
            track_ready.set_result(track)

    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    try:
        timeout = aiohttp.ClientTimeout(total=20.0)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            print("[probe] POSTing SDP offer ...")
            t0 = time.monotonic()
            async with http.post(
                transport_url,
                data=pc.localDescription.sdp,
                headers={"Accept": "application/sdp", "Content-Type": "application/sdp"},
                ssl=ssl_ctx,
                allow_redirects=True,
            ) as resp:
                body = await resp.text()
                print(f"[probe] POST -> HTTP {resp.status} in {time.monotonic()-t0:.2f}s, {len(body)} bytes SDP")
                if resp.status != 201:
                    print(f"[probe] !! expected 201, got {resp.status}. Body head:\n{body[:300]}")
                    return 2
                await pc.setRemoteDescription(RTCSessionDescription(sdp=body, type="answer"))
    except ssl.SSLError as e:
        print(f"[probe] !! SSL ERROR (cert not trusted / mismatch): {e}")
        return 3
    except aiohttp.ClientError as e:
        print(f"[probe] !! HTTP/connection error (POST never reached server?): {e!r}")
        return 4
    except Exception as e:
        print(f"[probe] !! POST failed: {type(e).__name__}: {e}")
        return 5

    # Report negotiated video codecs from the answer.
    codecs = [c.mimeType for t in pc.getTransceivers() if t.receiver
              for c in (t.receiver._codecs if hasattr(t.receiver, "_codecs") else [])]
    print(f"[probe] waiting up to {wait_s:.0f}s for a video track + frame ...")
    try:
        track = await asyncio.wait_for(track_ready, timeout=wait_s)
    except asyncio.TimeoutError:
        print("[probe] !! no video track within timeout — ICE/DTLS likely never connected.")
        print(f"[probe]    final ICE={pc.iceConnectionState} connection={pc.connectionState}")
        await pc.close()
        return 6

    try:
        frame = await asyncio.wait_for(track.recv(), timeout=wait_s)
        print(f"[probe] ✅ FIRST FRAME decoded: {getattr(frame,'width','?')}x{getattr(frame,'height','?')}")
        rc = 0
    except Exception as e:
        print(f"[probe] !! no decoded frame in {wait_s:.0f}s: {type(e).__name__}: {e}")
        rc = 7

    # RTP stats decide: were packets actually arriving?
    try:
        report = await pc.getStats()
        for stat in report.values():
            if getattr(stat, "type", "") == "inbound-rtp" and getattr(stat, "kind", "") == "video":
                pr = getattr(stat, "packetsReceived", "?")
                fd = getattr(stat, "framesDecoded", None)
                print(f"[probe] inbound-rtp video: packetsReceived={pr} framesDecoded={fd}")
                if pr in (0, "?", None) or pr == 0:
                    print("[probe]   => 0 packets: the phone is NOT sending video right now")
                    print("[probe]      (screen locked / browser backgrounded pauses the camera).")
                    print("[probe]      Keep the phone AWAKE with the publish page in the foreground.")
                elif rc == 7:
                    print("[probe]   => packets arrived but no frame decoded => decoder/keyframe issue")
    except Exception as e:
        print(f"[probe] (could not read stats: {e})")
    await pc.close()
    return rc


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url", help="webrtc+https://IP:8889/<path>/whep")
    ap.add_argument("--cert", help="self-signed cert to trust (diag/certs/server.crt)")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification")
    ap.add_argument("--wait", type=float, default=15.0)
    args = ap.parse_args(argv)
    return asyncio.run(probe(args.url, args.cert, args.wait, insecure=args.insecure))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
