"""Local pull selection and per-camera outgoing relays for the GPU gateway."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import anyio
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from .relay_auth import relay_headers

CONNECT = re.compile(r"/v1/(?:video/connect|fleet/cameras/(camera[0-9]+)/connect)$")
_CONNECT_TIMEOUT_SECONDS = 30

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
        headers = relay_headers(
            {
                key: value
                for key, value in (headers or {}).items()
                if key.lower() not in {"content-length", "content-encoding"}
            }
        )
        async with self.lock:
            # Cancellation cannot discard the response to a connection that
            # the remote runtime may already have accepted. Own negotiation
            # until its bounded result identifies whether rollback is safe.
            owner = asyncio.create_task(
                self._post_connect(client, path, payload, headers)
            )
            try:
                response = await asyncio.shield(owner)
            except BaseException:
                cancelled_during_cleanup = await self._finish_cleanup(
                    self._rollback_negotiation(owner, client, path, headers)
                )
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError from None
                raise
            if response.status_code != 200:
                return JSONResponse(response.json(), status_code=response.status_code)
            relay = None
            try:
                # Everything after remote acceptance owns a connection, even
                # malformed JSON or relay metadata from an older GPU host.
                result = response.json()
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
                relay = LocalVideoRelay(self.origin, socket_path, options)
                await relay.start()
                old = self.relays.get(camera)
                if old is not None:
                    await old.close()
            except BaseException as error:
                # Join cleanup even if the caller cancels again; never leave an
                # untracked receiver or an orphaned cleanup task behind.
                cancelled_during_cleanup = await self._finish_cleanup(
                    self._rollback_connect(client, path, headers, relay)
                )
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError from None
                if not isinstance(error, Exception) or isinstance(error, HTTPException):
                    raise
                raise HTTPException(
                    503, "Local video relay could not connect to GPU host"
                ) from None
            self.relays[camera] = relay
            return JSONResponse(result, headers={"Cache-Control": "no-store"})

    async def _post_connect(self, client, path, payload, headers):
        try:
            # The forwarding client's read timeout is intentionally unbounded
            # for streaming routes; connection negotiation has a total budget.
            return await asyncio.wait_for(
                client.post(
                    self.origin + path,
                    json={**payload, "relay": True},
                    headers=headers,
                    timeout=_CONNECT_TIMEOUT_SECONDS,
                ),
                timeout=_CONNECT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                503, "Local video relay could not connect to GPU host"
            ) from None

    async def _rollback_negotiation(self, owner, client, path, headers):
        try:
            response = await owner
        except (Exception, asyncio.CancelledError):
            return  # Unconfirmed acceptance must not disconnect another slot.
        if response.status_code == 200:
            await self._rollback_connect(client, path, headers, None)

    async def _finish_cleanup(self, operation):
        cancelled_during_cleanup = False
        with anyio.CancelScope(shield=True):
            cleanup = asyncio.create_task(operation)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled_during_cleanup = True
            cleanup.result()
        return cancelled_during_cleanup

    async def _rollback_connect(self, client, path, headers, relay):
        if relay is not None:
            try:
                await relay.close()
            except Exception:
                pass  # Local cleanup failure must not skip the remote rollback.
        try:
            await client.post(
                self.origin + path.removesuffix("connect") + "disconnect",
                json={},
                headers=headers,
                timeout=5,
            )
        except Exception:
            pass  # Preserve the original negotiation/start error without secrets.

    async def disconnect(self, path):
        async with self.lock:
            await self._disconnect_locked(path)

    async def _disconnect_locked(self, path: str) -> None:
        """Caller holds lock; retain the relay if local cleanup fails."""
        matched = DISCONNECT.fullmatch(path)
        camera = matched.group(1) or "camera01"
        relay = self.relays.get(camera)
        if relay is not None:
            await relay.close()
            self.relays.pop(camera, None)

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
