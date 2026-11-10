"""Read-only result subscriptions, including bounded fleet/gateway forwarding."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import urlsplit, urlunsplit

from fastapi import HTTPException, WebSocket
from starlette.websockets import WebSocketDisconnect

from .grounding import DETECTION_EVENT_LABELS

RESULT_SOCKET_PATHS = {
    "/results/ws": None,
    "/results/overall/ws": "overall",
    **{f"/results/{feature}/ws": feature for feature in DETECTION_EVENT_LABELS},
}
POLL_SECONDS = 0.1
HEARTBEAT_SECONDS = 15.0
SEND_TIMEOUT = 5.0
MESSAGE_LIMIT = 2 * 1024 * 1024


class SubscriptionError(Exception):
    def __init__(self, message, code=1008):
        super().__init__(message)
        self.code = code


def _encode(payload, camera_id=None):
    if camera_id is not None:
        from .fleet_outputs import scoped_payload

        payload = scoped_payload(payload, camera_id)
        if isinstance(payload.get("data"), dict):
            payload["data"]["camera_id"] = camera_id
    text = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    if len(text) > MESSAGE_LIMIT:
        raise SubscriptionError("result message exceeds 2 MiB", 1009)
    return text


async def _send(socket, payload, camera_id=None):
    text = await asyncio.to_thread(_encode, payload, camera_id)
    await asyncio.wait_for(socket.send_text(text), SEND_TIMEOUT)


async def _close(socket, code=1000):
    # Upstream empty/abnormal closes use synthetic codes forbidden on the wire.
    if code in {1005, 1006, 1015}:
        code = 1013
    try:
        await asyncio.wait_for(socket.close(code=code), SEND_TIMEOUT)
    except (RuntimeError, WebSocketDisconnect, OSError, TimeoutError):
        pass


async def reject_subscription(socket, message, code=1008, camera_id=None):
    try:
        await _send(socket, {"type": "error", "message": message}, camera_id)
    except (WebSocketDisconnect, OSError, TimeoutError):
        pass
    finally:
        await _close(socket, code)


async def _read_only(socket):
    message = await socket.receive()
    if message["type"] != "websocket.disconnect":
        raise SubscriptionError("result subscriptions do not accept commands")


async def _until_disconnect(socket, producer):
    tasks = [asyncio.create_task(producer), asyncio.create_task(_read_only(socket))]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _resume(socket):
    params = socket.query_params
    try:
        after = int(params.get("after", "0"))
        if after < 0:
            raise ValueError
        if after:
            return None, after
        token = params.get("last_event_id", socket.headers.get("last-event-id", ""))
        if not token:
            return None, 0
        epoch, separator, number = token.rpartition(":")
        if not separator or not epoch or len(epoch) > 128:
            raise ValueError
        cursor = int(number)
        if cursor < 0:
            raise ValueError
        return epoch, cursor
    except ValueError:
        raise SubscriptionError(
            "use after >= 0 or last_event_id=epoch:cursor",
        ) from None


async def stream_results(socket, runtime, channel=None, *, camera_id=None):
    """The caller accepts once; disconnecting never stops the analysis runtime."""
    from .service import _feature_result_payload, _overall_description_payload

    async def produce():
        epoch, current = _resume(socket)
        bus = runtime.results
        snapshot = await asyncio.to_thread(bus.snapshot_after, 0)
        await _send(
            socket,
            {
                "type": "hello",
                "protocol": 1,
                "epoch": snapshot.epoch,
                "latest_cursor": snapshot.cursor,
                "oldest_cursor": snapshot.oldest_cursor,
            },
            camera_id,
        )
        if epoch is None:
            epoch = snapshot.epoch
            if current == 0:
                # A fresh subscription starts at the retained history boundary.
                # Once connected, even cursor zero must detect later eviction.
                current = snapshot.oldest_cursor - 1
        loop = asyncio.get_running_loop()
        last_sent = loop.time()
        while True:
            reason = None
            if epoch != snapshot.epoch:
                reason = "stream_changed"
            elif current > snapshot.cursor:
                reason = "cursor_ahead"
            elif current < snapshot.oldest_cursor - 1:
                reason = "history_lost"
            if reason is not None:
                epoch = snapshot.epoch
                current = snapshot.oldest_cursor - 1
                await _send(
                    socket,
                    {
                        "type": "reset",
                        "reason": reason,
                        "id": f"{epoch}:{current}",
                        "oldest_cursor": snapshot.oldest_cursor,
                        "latest_cursor": snapshot.cursor,
                    },
                    camera_id,
                )
                last_sent = loop.time()
                # A new bus epoch can have lower cursors than the old one.
                snapshot = await asyncio.to_thread(bus.snapshot_after, current)
                continue
            for record in snapshot.records:
                if record.cursor <= current:
                    continue
                current = record.cursor
                payload = record.payload
                name = "analysis_result"
                if channel == "overall":
                    payload = _overall_description_payload(payload)
                    name = "overall_description"
                elif channel is not None:
                    payload = _feature_result_payload(payload, channel)
                    if payload is None:
                        continue
                await _send(
                    socket,
                    {
                        "type": name,
                        "id": f"{epoch}:{current}",
                        "data": payload,
                    },
                    camera_id,
                )
                last_sent = loop.time()
            if loop.time() - last_sent >= HEARTBEAT_SECONDS:
                await _send(
                    socket,
                    {
                        "type": "heartbeat",
                        "id": f"{epoch}:{current}",
                    },
                    camera_id,
                )
                last_sent = loop.time()
            await asyncio.sleep(POLL_SECONDS)
            snapshot = await asyncio.to_thread(bus.snapshot_after, current)

    try:
        await _until_disconnect(socket, produce())
    except SubscriptionError as error:
        await reject_subscription(socket, str(error), error.code, camera_id)
    except TimeoutError:
        await _close(socket, 1013)
    except (WebSocketDisconnect, OSError):
        pass
    finally:
        await _close(socket)


async def proxy_results(socket, origin, path, *, camera_id=None):
    """Only server-selected result paths/origins reach this read-only proxy."""
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed, InvalidHandshake

    parts = urlsplit(origin)
    url = urlunsplit(
        (
            "wss" if parts.scheme == "https" else "ws",
            parts.netloc,
            parts.path.rstrip("/") + path,
            socket.scope["query_string"].decode("ascii"),
            "",
        )
    )
    headers = {}
    if token := socket.headers.get("last-event-id"):
        headers["Last-Event-ID"] = token
    try:
        async with connect(
            url,
            additional_headers=headers,
            compression=None,
            max_size=MESSAGE_LIMIT,
            max_queue=4,
            open_timeout=5,
            close_timeout=2,
            proxy=None,
        ) as remote:

            async def download():
                async for message in remote:
                    if not isinstance(message, str):
                        raise SubscriptionError("invalid upstream result message", 1011)
                    payload = await asyncio.to_thread(json.loads, message)
                    if not isinstance(payload, dict):
                        raise SubscriptionError("invalid upstream result message", 1011)
                    await _send(socket, payload, camera_id)
                await _close(socket, remote.close_code or 1000)

            await _until_disconnect(socket, download())
    except SubscriptionError as error:
        await reject_subscription(socket, str(error), error.code, camera_id)
    except ConnectionClosed as error:
        close = error.rcvd or error.sent
        code = close.code if close is not None else 1013
        await _close(socket, code)
    except (OSError, TimeoutError, InvalidHandshake):
        await reject_subscription(socket, "result service unavailable", 1013, camera_id)
    except (ValueError, TypeError):
        await reject_subscription(
            socket,
            "invalid upstream result message",
            1011,
            camera_id,
        )
    except WebSocketDisconnect:
        pass
    finally:
        await _close(socket)


def install_result_socket_routes(application):
    def endpoint(channel):
        async def subscribe(socket: WebSocket):
            await socket.accept()
            await stream_results(socket, socket.app.state.runtime, channel)

        return subscribe

    for suffix, channel in RESULT_SOCKET_PATHS.items():
        application.add_api_websocket_route("/v1" + suffix, endpoint(channel))


def install_fleet_result_socket_routes(application, controller):
    from .fleet_outputs import PREFIX

    async def subscribe(socket: WebSocket):
        await socket.accept()
        camera_id = socket.path_params["camera_id"]
        try:
            index = controller.resolve_camera(camera_id)
            controller._require_primary()
        except HTTPException:
            await reject_subscription(socket, "camera output not available")
            return
        suffix = socket.scope["path"][len(PREFIX.format(camera_id=camera_id)) :]
        if index == controller.camera_index:
            await stream_results(
                socket,
                application.state.runtime,
                RESULT_SOCKET_PATHS[suffix],
                camera_id=camera_id,
            )
        else:
            await proxy_results(
                socket,
                controller._url(index, ""),
                "/v1" + suffix,
                camera_id=camera_id,
            )

    for suffix in RESULT_SOCKET_PATHS:
        application.add_api_websocket_route(PREFIX + suffix, subscribe)
