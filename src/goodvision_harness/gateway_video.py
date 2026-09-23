"""Local pull selection and per-camera outgoing relays for the GPU gateway."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

CONNECT = re.compile(r"/v1/(?:video/connect|fleet/cameras/(camera[0-9]+)/connect)$")
DISCONNECT = re.compile(
    r"/v1/(?:video/disconnect|fleet/cameras/(camera[0-9]+)/disconnect)$"
)


class GatewayVideo:
    def __init__(self, origin, *, settings_path=None):
        self.origin = origin
        self.relays = {}
        self.lock = asyncio.Lock()
        self.path = Path(
            settings_path
            or os.environ.get(
                "HARNESS_GATEWAY_SETTINGS_PATH",
                "/data/gateway-settings.json",
            )
        )
        self.location = os.environ.get("HARNESS_VIDEO_PULL_LOCATION", "local")
        if self.path.exists():
            self.location = json.loads(self.path.read_text("utf-8"))["pull_location"]
        if self.location not in {"local", "gpu"}:
            raise ValueError("HARNESS_VIDEO_PULL_LOCATION must be local or gpu")

    def settings(self):
        return {"pull_location": self.location, "applies_to": "next_connection"}

    def save(self, payload):
        if (
            not isinstance(payload, dict)
            or set(payload) != {"pull_location"}
            or payload["pull_location"] not in {"local", "gpu"}
        ):
            raise ValueError("pull_location must be local or gpu")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), "utf-8")
        temporary.replace(self.path)
        self.location = payload["pull_location"]
        return self.settings()

    async def close(self):
        await asyncio.gather(*(relay.close() for relay in self.relays.values()))
        self.relays.clear()

    async def connect(self, client, path, payload, headers=None):
        from .video_relay import LocalVideoRelay

        matched = CONNECT.fullmatch(path)
        camera = matched.group(1) or "camera01"
        headers = {
            key: value
            for key, value in (headers or {}).items()
            if key.lower() not in {"content-length", "content-encoding"}
        }
        async with self.lock:
            response = await client.post(
                self.origin + path,
                json={**payload, "relay": True},
                headers=headers,
            )
            result = response.json()
            if response.status_code != 200:
                return JSONResponse(result, status_code=response.status_code)
            options = result.pop("video_relay", None)
            if not isinstance(options, dict):
                raise HTTPException(
                    503,
                    "GPU host does not support local video ingress; "
                    "update its analysis service",
                )
            socket_path = (
                f"/v1/fleet/cameras/{camera}/video/relay/{options['id']}"
                if matched.group(1)
                else f"/v1/video/relay/{options['id']}"
            )
            relay = None
            try:
                relay = LocalVideoRelay(self.origin, socket_path, options)
                await relay.start()
            except Exception:
                if relay is not None:
                    await relay.close()
                try:
                    await client.post(
                        self.origin + path.removesuffix("connect") + "disconnect",
                        json={},
                        headers=headers,
                    )
                except httpx.HTTPError:
                    pass
                raise HTTPException(
                    503, "Local video relay could not connect to GPU host"
                ) from None
            old = self.relays.pop(camera, None)
            if old is not None:
                await old.close()
            self.relays[camera] = relay
            return JSONResponse(result, headers={"Cache-Control": "no-store"})

    async def disconnect(self, path):
        matched = DISCONNECT.fullmatch(path)
        camera = matched.group(1) or "camera01"
        async with self.lock:
            relay = self.relays.pop(camera, None)
            if relay is not None:
                await relay.close()

    async def prepare_legacy_start(self, client, payload, headers=None):
        relay = self.relays.get("camera01")
        if relay is None or relay.task.done():
            response = await self.connect(
                client,
                "/v1/video/connect",
                {
                    key: payload[key]
                    for key in ("stream_url", "camera_layout")
                    if key in payload
                },
                headers,
            )
            if response.status_code != 200:
                return response
            relay = self.relays["camera01"]
        first = asyncio.create_task(relay.first_frame.wait())
        try:
            await asyncio.wait(
                [first, relay.task], timeout=30, return_when=asyncio.FIRST_COMPLETED
            )
            if not relay.first_frame.is_set() or relay.task.done():
                raise HTTPException(503, "Local camera is not delivering video")
        finally:
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
        return None
