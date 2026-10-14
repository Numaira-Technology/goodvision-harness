"""Persistent event resources, separate from the immediate analysis stream."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send

from .privacy import public_safe_value


async def _archive(request: Request) -> Any:
    try:
        return await asyncio.to_thread(request.app.state.runtime.event_archive)
    except RuntimeError:
        raise HTTPException(503, "event archive unavailable") from None


async def _read(method, *args, **kwargs) -> Any:
    try:
        return await asyncio.to_thread(method, *args, **kwargs)
    except RuntimeError:
        raise HTTPException(503, "event archive unavailable") from None


class _LeasedVideoResponse(FileResponse):
    def __init__(self, path, *, release: Callable[[], None], **kwargs):
        super().__init__(path, **kwargs)
        self._release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await asyncio.shield(asyncio.to_thread(self._release))


def install_event_routes(application: FastAPI) -> None:
    @application.get("/v1/events", tags=["Event evidence"])
    async def event_list(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        before: str | None = None,
        stream_session_id: str | None = None,
        event_label: str | None = None,
        subject_id: str | None = None,
        message_id: str | None = None,
        since: float | None = Query(None, ge=0),
        until: float | None = Query(None, ge=0),
    ) -> JSONResponse:
        archive = await _archive(request)
        if since is not None and until is not None and since > until:
            raise HTTPException(400, "since must not exceed until")
        try:
            payload = await _read(
                archive.list_events,
                limit=limit,
                before=before,
                stream_session_id=stream_session_id,
                event_label=event_label,
                subject_id=subject_id,
                message_id=message_id,
                since=since,
                until=until,
            )
        except (TypeError, ValueError):
            raise HTTPException(400, "invalid event query") from None
        return JSONResponse(
            public_safe_value(payload), headers={"Cache-Control": "no-store"}
        )

    # The fixed stream path must precede the variable event_id path.
    @application.get(
        "/v1/events/stream",
        tags=["Event evidence"],
        response_class=StreamingResponse,
        responses={
            200: {"content": {"text/event-stream": {}}},
            503: {"description": "Event archive unavailable"},
        },
    )
    async def event_stream(
        request: Request, after: str | None = None
    ) -> StreamingResponse:
        archive = await _archive(request)
        cursor = request.headers.get("last-event-id") or after
        initial = await _read(archive.changes_after, cursor)

        async def messages():
            nonlocal cursor
            batch = initial
            heartbeat = time.monotonic()
            while not await request.is_disconnected():
                epoch = batch.get("epoch", "")
                next_cursor = batch.get("cursor", 0)
                reset_reason = batch.get("reset_reason")
                if reset_reason:
                    cursor = f"{epoch}:{next_cursor}"
                    payload = {"reason": reset_reason, "cursor": cursor}
                    yield f"id: {cursor}\nevent: reset\ndata: {json.dumps(payload)}\n\n"
                else:
                    for change in batch.get("changes", []):
                        change_cursor = change.get(
                            "cursor", change.get("sequence", next_cursor)
                        )
                        cursor = f"{epoch}:{change_cursor}"
                        payload = public_safe_value(change)
                        yield (
                            f"id: {cursor}\nevent: event_change\ndata: "
                            + json.dumps(payload, ensure_ascii=False, allow_nan=False)
                            + "\n\n"
                        )
                    if cursor is None:
                        cursor = f"{epoch}:{next_cursor}"
                if time.monotonic() - heartbeat >= 10:
                    yield ": keep-alive\n\n"
                    heartbeat = time.monotonic()
                await asyncio.sleep(0.25)
                try:
                    batch = await asyncio.to_thread(archive.changes_after, cursor)
                except RuntimeError:
                    yield 'event: reset\ndata: {"reason":"archive_unavailable"}\n\n'
                    return

        return StreamingResponse(
            messages(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @application.get("/v1/events/{event_id}", tags=["Event evidence"])
    async def event_detail(event_id: str, request: Request) -> JSONResponse:
        archive = await _archive(request)
        detail = await _read(archive.get_event, event_id)
        if detail is None:
            raise HTTPException(404, "event not found")
        return JSONResponse(
            public_safe_value(detail), headers={"Cache-Control": "no-store"}
        )

    @application.get(
        "/v1/events/{event_id}/video",
        tags=["Event evidence"],
        response_class=FileResponse,
        response_model=None,
        responses={
            200: {"description": "Playable MP4", "content": {"video/mp4": {}}},
            206: {"description": "Byte range", "content": {"video/mp4": {}}},
            202: {
                "description": "Evidence pending",
                "content": {"application/json": {}},
            },
            404: {"description": "Event or immutable video revision not found"},
            416: {"description": "Requested byte range cannot be satisfied"},
            503: {"description": "Evidence failed or archive unavailable"},
        },
    )
    async def event_video(
        event_id: str,
        request: Request,
        revision: int | None = Query(None, ge=1),
    ) -> FileResponse | JSONResponse:
        archive = await _archive(request)
        detail = await _read(archive.get_event, event_id)
        if detail is None:
            raise HTTPException(404, "event not found")
        lease = await _read(archive.acquire_video, event_id, revision=revision)
        if lease is None:
            if revision is not None:
                raise HTTPException(404, "video revision not found")
            evidence = detail.get("evidence", {})
            status = 503 if evidence.get("status") in {"failed", "expired"} else 202
            return JSONResponse(
                public_safe_value(
                    {"event_id": detail["event_id"], "evidence": evidence}
                ),
                status_code=status,
                headers={"Cache-Control": "no-store", "Retry-After": "1"},
            )
        # Keep the immutable file alive from lookup through playback completion.
        path, release = lease
        return _LeasedVideoResponse(
            path,
            release=release,
            media_type="video/mp4",
            headers={
                "Cache-Control": "no-store",
                "X-Event-Id": detail["event_id"],
            },
        )
