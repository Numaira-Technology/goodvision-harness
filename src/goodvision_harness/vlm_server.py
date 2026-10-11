"""Local VLM companion: bounded CPU preparation and OpenAI API forwarding."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from .model_api import ModelAPIError, ModelAPISettings, model_request_headers
from .scene_model import TemporalVLMInspector, _json
from .vlm_compact_model import CompactGroundingInspector, EventJSONInspector
from .vlm_independent_model import IndependentVLMInspector
from .vlm_lean_grounding import LeanGroundingInspector
from .vlm_offload import MAX_COMPLETION_BYTES, MAX_REQUEST_BYTES, decode_inspection

_MAX_STREAM_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class VLMServerSettings:
    upstream_endpoint: str = "http://vlm-server:8000/v1/chat/completions"
    model: str = "local-vision"
    api_key: str = field(default="", repr=False)
    max_concurrency: int = 8
    preparation_concurrency: int = 2
    timeout_seconds: float = 90.0

    def __post_init__(self):
        parsed = urlsplit(self.upstream_endpoint)
        try:
            port = parsed.port
        except ValueError:
            raise ValueError("invalid VLM upstream endpoint") from None
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith("/chat/completions")
            or any(c.isspace() for c in self.upstream_endpoint)
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise ValueError("invalid VLM upstream endpoint")
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or len(self.model) > 512
            or any(ord(c) < 32 or ord(c) == 127 for c in self.model)
        ):
            raise ValueError("invalid VLM served model")
        for value in (self.max_concurrency, self.preparation_concurrency):
            if type(value) is not int or not 1 <= value <= 64:
                raise ValueError("invalid VLM concurrency limit")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 300
        ):
            raise ValueError("invalid VLM deadline")
        if not isinstance(self.api_key, str):
            raise ValueError("invalid VLM local credential")

    @classmethod
    def from_environment(cls):
        return cls(
            upstream_endpoint=os.environ.get(
                "VLM_SERVER_UPSTREAM_ENDPOINT",
                "http://vlm-server:8000/v1/chat/completions",
            ),
            model=os.environ.get(
                "VLM_SERVER_MODEL",
                os.environ.get("LOCAL_INFERENCE_MODEL", "local-vision"),
            ),
            api_key=os.environ.get("LOCAL_INFERENCE_API_KEY", "").strip(),
            max_concurrency=int(os.environ.get("VLM_SERVER_MAX_CONCURRENCY", "8")),
            preparation_concurrency=int(
                os.environ.get("VLM_SERVER_PREPARATION_CONCURRENCY", "2")
            ),
            timeout_seconds=float(os.environ.get("VLM_SERVER_TIMEOUT_SECONDS", "90")),
        )

    @property
    def models_endpoint(self):
        parts = urlsplit(self.upstream_endpoint)
        path = parts.path.removesuffix("/chat/completions") + "/models"
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


class _Admission:
    """Accessed on the ASGI event loop; threads never release this gate."""

    def __init__(self, capacity):
        self.capacity, self.active = capacity, 0

    def acquire(self):
        if self.active >= self.capacity:
            raise HTTPException(
                429, "VLM capacity exhausted", headers={"Retry-After": "1"}
            )
        self.active += 1

    def release(self):
        self.active -= 1


class _Execution:
    def __init__(self, started, timeout, loop):
        self.started = started
        self.deadline = started + timeout
        self.loop = loop
        self.metadata_ready = asyncio.Event()
        self.abandoned = threading.Event()
        self.inspector = None

    def set_deadline(self, timeout):
        self.deadline = min(self.deadline, self.started + timeout)
        self.loop.call_soon_threadsafe(self.metadata_ready.set)

    def cancel(self):
        self.abandoned.set()
        inspector = self.inspector
        if inspector is not None:
            inspector.cancel_request()

    def check(self):
        if self.abandoned.is_set() or time.monotonic() >= self.deadline:
            raise ModelAPIError("VLM deadline exceeded", code="timeout")


class _CaptureMixin:
    def __init__(self, *args, execution, preparation, client_factory, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution = execution
        self.preparation = preparation
        self.client_factory = client_factory
        self.completion = None
        self._active_request = None

    def _ensure_transport(self):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        if self._http is None:
            self._http = self.client_factory()

    def build_messages(self, job, *, encoded_images=None):
        self.execution.check()
        remaining = max(0, self.execution.deadline - time.monotonic())
        if not self.preparation.acquire(timeout=remaining):
            raise ModelAPIError("VLM deadline exceeded", code="timeout")
        try:
            self.execution.check()
            result = super().build_messages(job, encoded_images=encoded_images)
            self.execution.check()
            return result
        finally:
            self.preparation.release()

    async def _request(self, payload, headers, *, timing):
        self.execution.check()
        self._active_request = asyncio.current_task()
        try:
            remaining = self.execution.deadline - time.monotonic()
            self.completion = await asyncio.wait_for(
                super()._request(payload, headers, timing=timing),
                remaining,
            )
            return self.completion
        finally:
            self._active_request = None

    def cancel_request(self):
        loop, task = self._loop, self._active_request
        if loop is not None and task is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass


class _CapturedTemporal(_CaptureMixin, TemporalVLMInspector):
    pass


class _CapturedFull(_CaptureMixin, IndependentVLMInspector):
    pass


class _CapturedCompact(_CaptureMixin, CompactGroundingInspector):
    pass


class _CapturedEvents(_CaptureMixin, EventJSONInspector):
    pass


class _CapturedLean(_CaptureMixin, LeanGroundingInspector):
    pass


_INSPECTORS = {
    "temporal": _CapturedTemporal,
    "full": _CapturedFull,
    "compact": _CapturedCompact,
    "events": _CapturedEvents,
    "lean-v1": _CapturedLean,
}


def _error(status, code):
    return JSONResponse(
        {"code": code, "detail": "VLM request failed"},
        status_code=status,
        headers={"Retry-After": "1"} if status == 429 else None,
    )


def _numeric_timing(inspector):
    result = {}
    for name in ("request_prepare_seconds", "verdict_parse_seconds", "total_seconds"):
        value = inspector.last_inspection_timing.get(name)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            result[name] = value
    return result


def _safe_usage(completion):
    usage = completion.get("usage")
    if not isinstance(usage, dict):
        return {}
    result = {
        name: usage[name]
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if type(usage.get(name)) is int and usage[name] >= 0
    }
    for name, field_name in (
        ("completion_tokens_details", "reasoning_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    ):
        detail = usage.get(name)
        if (
            isinstance(detail, dict)
            and type(detail.get(field_name)) is int
            and detail[field_name] >= 0
        ):
            result[name] = {field_name: detail[field_name]}
    return result


def _run_inspection(body, settings, execution, preparation, client_factory):
    inspector = None
    try:
        execution.check()
        decoded = decode_inspection(_json(body, stage="request"))
        if decoded.model != settings.model:
            raise ValueError("unknown served model")
        execution.set_deadline(decoded.deadline_seconds)
        execution.check()
        model_settings = ModelAPISettings(
            provider="local",
            endpoint=settings.upstream_endpoint,
            timeout_seconds=min(settings.timeout_seconds, decoded.deadline_seconds),
            max_retries=0,
        )
        inspector = _INSPECTORS[decoded.kind](
            model_settings,
            model=settings.model,
            api_key=settings.api_key,
            request_parameters=decoded.parameters,
            **decoded.options,
            execution=execution,
            preparation=preparation,
            client_factory=client_factory,
        )
        execution.inspector = inspector
        inspector.inspect(decoded.job)
        execution.check()
        completion = inspector.completion
        if not isinstance(completion, dict):
            raise ModelAPIError("Invalid VLM completion", code="invalid_response")
        envelope = {
            "version": 1,
            "request_id": decoded.job.request.request_id,
            "completion": completion,
            "timing": _numeric_timing(inspector),
            "usage": _safe_usage(completion),
        }
        response = json.dumps(envelope, allow_nan=False, separators=(",", ":")).encode()
        if len(response) > MAX_COMPLETION_BYTES:
            raise ModelAPIError("Invalid VLM completion", code="invalid_response")
        return response
    except asyncio.CancelledError:
        raise ModelAPIError("VLM deadline exceeded", code="timeout") from None
    finally:
        execution.inspector = None
        if inspector is not None:
            inspector.close()


async def _receive(request, deadline):
    length = request.headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
        except ValueError:
            raise HTTPException(400, "Invalid request length") from None
        if size < 0 or size > MAX_REQUEST_BYTES:
            raise HTTPException(413, "VLM request exceeds byte limit")
    body = bytearray()
    iterator = request.stream().__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(
                iterator.__anext__(), max(0, deadline - time.monotonic())
            )
        except StopAsyncIteration:
            return bytes(body)
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise HTTPException(413, "VLM request exceeds byte limit")
        body.extend(chunk)


async def _read_response(response, limit, deadline):
    body = bytearray()
    iterator = response.aiter_bytes().__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(
                iterator.__anext__(), max(0, deadline - time.monotonic())
            )
        except StopAsyncIteration:
            return bytes(body)
        if len(body) + len(chunk) > limit:
            raise ModelAPIError("Invalid VLM response", code="invalid_response")
        body.extend(chunk)


async def _watch_disconnect(request):
    while True:
        if (await request.receive())["type"] == "http.disconnect":
            return


async def _await_work(future, disconnect, deadline):
    completed, _ = await asyncio.wait(
        {future, disconnect},
        return_when=asyncio.FIRST_COMPLETED,
        timeout=max(0, deadline - time.monotonic()),
    )
    if future in completed:
        return await future
    if disconnect in completed:
        raise ClientDisconnect()
    raise TimeoutError()


class _OwnedStreamingResponse(StreamingResponse):
    """Release upstream ownership even if the body iterator never starts."""

    def __init__(self, *args, cleanup, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.cleanup()


def create_app(settings=None, *, client_factory=None):
    settings = settings or VLMServerSettings.from_environment()
    if client_factory is None:

        def client_factory():
            return httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                timeout=settings.timeout_seconds,
                limits=httpx.Limits(
                    max_connections=settings.max_concurrency,
                    max_keepalive_connections=settings.max_concurrency,
                ),
            )

    admission = _Admission(settings.max_concurrency)
    preparation = threading.BoundedSemaphore(settings.preparation_concurrency)

    @asynccontextmanager
    async def lifespan(app):
        app.state.executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrency, thread_name_prefix="vlm-companion"
        )
        app.state.http = client_factory()
        try:
            yield
        finally:
            await asyncio.to_thread(
                app.state.executor.shutdown, wait=True, cancel_futures=True
            )
            await app.state.http.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.admission = admission

    def authorize(request):
        if settings.api_key and not hmac.compare_digest(
            request.headers.get("authorization", "").encode(),
            ("Bearer " + settings.api_key).encode(),
        ):
            raise HTTPException(401, "VLM authentication required")

    def upstream_request(method, endpoint, *, payload=None, timeout=3):
        local = ModelAPISettings(provider="local", endpoint=settings.upstream_endpoint)
        # Fresh Request objects never inherit pool cookies, auth or query params.
        return httpx.Request(
            method,
            endpoint,
            headers=model_request_headers(local, settings.api_key),
            json=payload,
            extensions={"timeout": httpx.Timeout(timeout).as_dict()},
        )

    @app.get("/health")
    async def health():
        ready = False
        response = None
        try:
            request = upstream_request(
                "GET",
                settings.models_endpoint,
                timeout=min(3, settings.timeout_seconds),
            )
            response = await asyncio.wait_for(
                app.state.http.send(
                    request, stream=True, auth=None, follow_redirects=False
                ),
                3,
            )
            if response.status_code == 200:
                body = await _read_response(response, 64 * 1024, time.monotonic() + 3)
                payload = _json(body, stage="models")
                ready = (
                    isinstance(payload, dict)
                    and isinstance(payload.get("data"), list)
                    and any(
                        isinstance(item, dict) and item.get("id") == settings.model
                        for item in payload["data"]
                    )
                )
        except (
            ModelAPIError,
            httpx.HTTPError,
            TimeoutError,
            ValueError,
            TypeError,
            RecursionError,
        ):
            pass
        finally:
            if response is not None:
                await response.aclose()
        return JSONResponse(
            {"ready": ready, "model": settings.model, "protocol": "goodvision"},
            status_code=200 if ready else 503,
        )

    @app.post("/v1/inspections")
    async def inspections(request: Request):
        authorize(request)
        admission.acquire()
        loop = asyncio.get_running_loop()
        execution = _Execution(time.monotonic(), settings.timeout_seconds, loop)
        future = metadata = disconnect = None
        try:
            body = await _receive(request, execution.deadline)
            future = loop.run_in_executor(
                app.state.executor,
                _run_inspection,
                body,
                settings,
                execution,
                preparation,
                client_factory,
            )

            def finished(done):
                admission.release()
                if not done.cancelled():
                    done.exception()

            future.add_done_callback(finished)
            metadata = asyncio.create_task(execution.metadata_ready.wait())
            disconnect = asyncio.create_task(_watch_disconnect(request))
            completed, _ = await asyncio.wait(
                {future, metadata, disconnect},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=max(0, execution.deadline - time.monotonic()),
            )
            if not completed:
                raise TimeoutError()
            if disconnect in completed:
                raise ClientDisconnect()
            result = await _await_work(future, disconnect, execution.deadline)
            return Response(result, media_type="application/json")
        except (TimeoutError, httpx.TimeoutException):
            execution.cancel()
            return _error(504, "timeout")
        except asyncio.CancelledError:
            execution.cancel()
            raise
        except ClientDisconnect:
            execution.cancel()
            return Response(status_code=499)
        except ModelAPIError as error:
            if error.code == "timeout":
                return _error(504, "timeout")
            if error.status_code == 429 or error.code == "overloaded":
                return _error(429, "overloaded")
            return _error(
                502,
                error.code
                if error.code in {"invalid_response", "empty_response", "connection"}
                else "request_failed",
            )
        except (ValueError, TypeError, KeyError, RecursionError):
            return _error(400, "invalid_response")
        finally:
            if metadata is not None:
                metadata.cancel()
            if disconnect is not None:
                disconnect.cancel()
            if future is None:
                admission.release()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        authorize(request)
        admission.acquire()
        response = None
        transferred = False
        disconnect = upstream_task = None
        deadline = time.monotonic() + settings.timeout_seconds
        try:
            body = await _receive(request, deadline)
            payload = _json(body, stage="request")
            if (
                not isinstance(payload, dict)
                or payload.get("model") != settings.model
                or not isinstance(payload.get("messages"), list)
                or not payload["messages"]
                or {"endpoint", "api_key", "base_url"} & payload.keys()
                or type(payload.get("stream", False)) is not bool
            ):
                return _error(400, "invalid_response")
            for name in ("max_tokens", "max_completion_tokens"):
                if name in payload and (
                    type(payload[name]) is not int or not 1 <= payload[name] <= 16384
                ):
                    return _error(400, "invalid_response")
            if not {"max_tokens", "max_completion_tokens"} & payload.keys():
                payload["max_tokens"] = 1600
            upstream = upstream_request(
                "POST",
                settings.upstream_endpoint,
                payload=payload,
                timeout=max(0.001, deadline - time.monotonic()),
            )
            disconnect = asyncio.create_task(_watch_disconnect(request))
            upstream_task = asyncio.create_task(
                app.state.http.send(
                    upstream, stream=True, auth=None, follow_redirects=False
                )
            )
            response = await _await_work(upstream_task, disconnect, deadline)
            if not 200 <= response.status_code < 300:
                return _error(
                    429 if response.status_code == 429 else 502,
                    "overloaded" if response.status_code == 429 else "request_failed",
                )
            if payload.get("stream") is True:
                # StreamingResponse takes over receive/disconnect ownership.
                disconnect.cancel()
                await asyncio.gather(disconnect, return_exceptions=True)
                disconnect = None
                released = False

                async def cleanup():
                    nonlocal released
                    if released:
                        return
                    released = True
                    # Disconnect cancellation must not interrupt upstream close
                    # and release its capacity before the connection is closed.
                    with anyio.CancelScope(shield=True):
                        try:
                            await response.aclose()
                        finally:
                            admission.release()

                async def stream():
                    total = 0
                    try:
                        iterator = response.aiter_bytes().__aiter__()
                        while True:
                            try:
                                chunk = await asyncio.wait_for(
                                    iterator.__anext__(),
                                    max(0, deadline - time.monotonic()),
                                )
                            except StopAsyncIteration:
                                return
                            total += len(chunk)
                            if total > _MAX_STREAM_BYTES:
                                return
                            yield chunk
                    except (httpx.HTTPError, TimeoutError):
                        return
                    finally:
                        await cleanup()

                transferred = True
                return _OwnedStreamingResponse(
                    stream(),
                    cleanup=cleanup,
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"},
                )
            upstream_task = asyncio.create_task(
                _read_response(response, MAX_COMPLETION_BYTES, deadline)
            )
            result = await _await_work(upstream_task, disconnect, deadline)
            try:
                if not isinstance(_json(result, stage="completion"), dict):
                    return _error(502, "invalid_response")
            except ValueError:
                return _error(502, "invalid_response")
            return Response(result, media_type="application/json")
        except (TimeoutError, httpx.TimeoutException):
            return _error(504, "timeout")
        except ClientDisconnect:
            return Response(status_code=499)
        except httpx.HTTPError:
            return _error(502, "connection")
        except ModelAPIError:
            return _error(502, "invalid_response")
        except (ValueError, TypeError, RecursionError):
            return _error(400, "invalid_response")
        finally:
            if disconnect is not None:
                disconnect.cancel()
            if upstream_task is not None and not upstream_task.done():
                upstream_task.cancel()
                await asyncio.gather(upstream_task, return_exceptions=True)
            if not transferred:
                try:
                    if response is not None:
                        await response.aclose()
                finally:
                    admission.release()

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
