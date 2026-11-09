"""Standalone CUDA classifier. Install transformers/torch separately to serve it."""

import argparse
import asyncio
import base64
import binascii
import io
import json
import math
import queue
import threading
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

MODEL = "openai/clip-vit-large-patch14"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
PROMPTS = {
    "fight": (
        "People are physically fighting, punching or kicking each other.",
        "People are walking or talking peacefully.",
    ),
    "smoking": (
        "A person is smoking a cigarette or vaping.",
        "A person is eating, drinking or touching their face.",
    ),
    "fall": (
        "A person is losing balance and falling to the ground.",
        "A person is standing, walking or sitting steadily.",
    ),
    "weapon": (
        "A person is holding a gun or a knife.",
        "A person is holding a phone or ordinary object.",
    ),
    "ppe_violation": (
        "A worker in a hazardous work area is missing required protective equipment.",
        "Workers are wearing required protective equipment.",
    ),
    "ground_litter": (
        "Discarded rubbish is lying on the ground.",
        "The floor and ground are clean.",
    ),
}


class Encoder:
    def __init__(self, cache_dir):
        import torch
        from transformers import AutoModel, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA GPU is required")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            MODEL, revision=REVISION, cache_dir=cache_dir
        )
        self.model = (
            AutoModel.from_pretrained(MODEL, revision=REVISION, cache_dir=cache_dir)
            .eval()
            .to("cuda")
        )
        self._text_features = None

    def warmup(self):
        # Loading weights alone does not prove compatible CUDA kernels or the
        # processor/model interface. Do not publish ready until inference works.
        labels = list(PROMPTS)
        pairs = self.scores(Image.new("RGB", (224, 224)), labels)
        if len(pairs) != len(labels) or not all(_valid_pair(pair) for pair in pairs):
            raise ValueError("invalid model scores during warmup")

    def scores(self, image, labels):
        return self.score_batch([self.prepare(image)], [labels])[0]

    def prepare(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"]

    def score_batch(self, pixels, labels):
        torch = self.torch
        counts = [item.shape[0] for item in pixels]
        with torch.inference_mode():
            if self._text_features is None:
                inputs = self.processor(
                    text=[text for pair in PROMPTS.values() for text in pair],
                    return_tensors="pt", padding=True, truncation=True,
                )
                pooled = self.model.text_model(
                    **{key: value.to("cuda") for key, value in inputs.items()}
                ).pooler_output
                text = self.model.text_projection(pooled)
                self._text_features = text / text.norm(dim=-1, keepdim=True)
            pooled = self.model.vision_model(
                pixel_values=torch.cat(pixels).to("cuda")
            ).pooler_output
            images = self.model.visual_projection(pooled)
            images = images / images.norm(dim=-1, keepdim=True)
            logits = (
                self.model.logit_scale.exp()
                * (self._text_features @ images.t()).t()
            )
            pairs = logits.reshape(-1, len(PROMPTS), 2).softmax(dim=-1)
            pairs = pairs.cpu().tolist()
        indices = {label: index for index, label in enumerate(PROMPTS)}
        result, offset = [], 0
        for count, requested in zip(counts, labels, strict=True):
            window = _window_pairs(pairs[offset:offset + count])
            result.append([window[indices[label]] for label in requested])
            offset += count
        return result


class _BatchWorker:
    """Score up to four already-prepared requests without waiting to fill a batch."""

    def __init__(self, score_batch, max_batch_size):
        self.score_batch, self.max_batch_size = score_batch, max_batch_size
        self.pending = queue.Queue()
        self.thread = threading.Thread(target=self.run, name="clip-inference", daemon=True)

    def submit(self, pixels, labels, finish):
        result = Future()
        self.pending.put((result, pixels, labels, finish, time.perf_counter()))
        return result

    def run(self):
        while True:
            first = self.pending.get()
            if first is None:
                return
            batch = [first]
            stopping = False
            while len(batch) < self.max_batch_size:
                try:
                    item = self.pending.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    stopping = True
                    break
                batch.append(item)
            batch = [item for item in batch if item[0].set_running_or_notify_cancel()]
            if batch:
                started = time.perf_counter()
                error, pairs = None, [None] * len(batch)
                try:
                    pairs = self.score_batch(
                        [item[1] for item in batch], [item[2] for item in batch]
                    )
                except Exception as failure:
                    error = failure
                elapsed = time.perf_counter() - started
                for index, item in enumerate(batch):
                    future, _pixels, _labels, finish, queued = item
                    try:
                        response = finish(pairs[index], error, {
                            "server_queue_seconds": started - queued,
                            "model_pipeline_seconds": elapsed,
                            "batch_size": len(batch),
                        })
                        future.set_result(response)
                    except Exception as failure:
                        future.set_exception(failure)
            if stopping:
                return


def _valid_pair(pair):
    return (
        len(pair) == 2
        and all(math.isfinite(p) and 0 <= p <= 1 for p in pair)
        and abs(sum(pair) - 1) <= 1e-5
    )


def _window_pairs(frame_pairs):
    """An event is present if any supplied frame supports it."""
    result = []
    for pairs in zip(*frame_pairs, strict=True):
        invalid = next((pair for pair in pairs if not _valid_pair(pair)), None)
        result.append(
            invalid if invalid is not None else max(pairs, key=lambda p: p[0])
        )
    return result


def create_app(encoder, *, max_batch_size=4):
    ppe, shared_error = None, None

    def score_batch(pixels, labels):
        if shared_error is not None:
            raise HTTPException(503, "recognition service CUDA unavailable")
        return encoder.score_batch(pixels, labels)

    worker = _BatchWorker(score_batch, max_batch_size)

    @asynccontextmanager
    async def lifespan(_app):
        worker.thread.start()
        try:
            yield
        finally:
            while True:
                try:
                    item = worker.pending.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    item[0].cancel()
            worker.pending.put(None)
            worker.thread.join()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ready": shared_error is None, "model": MODEL}

    @app.post("/predict")
    async def predict(request: Request):
        received = time.perf_counter()
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 24 * 1024 * 1024:
                raise HTTPException(413, "request too large")
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("invalid envelope")
            images, questions = payload["images"], payload["question_template"]
            if not isinstance(images, list) or not 1 <= len(images) <= 4:
                raise ValueError("invalid image count")
            if not isinstance(questions, list) or not 1 <= len(questions) <= len(
                PROMPTS
            ):
                raise ValueError("unsupported questions")
            if any(
                not isinstance(q, dict)
                or not isinstance(q.get("id"), str)
                or not q["id"].startswith("scene/")
                for q in questions
            ):
                raise ValueError("invalid question")
            labels = [q["id"].removeprefix("scene/") for q in questions]
            if len(set(labels)) != len(labels) or any(
                label not in PROMPTS for label in labels
            ):
                raise ValueError("unsupported events")
            from .videojev_core import OPTIONS

            if any(
                q.get("task") != "choice" or q.get("options") != list(OPTIONS)
                for q in questions
            ):
                raise ValueError("invalid choices")
            if (
                not isinstance(payload["request_id"], str)
                or not 1 <= len(payload["request_id"]) <= 256
            ):
                raise ValueError("invalid request id")
            def prepare():
                decoded = []
                for encoded in images:
                    with Image.open(
                        io.BytesIO(base64.b64decode(encoded, validate=True))
                    ) as source:
                        if source.format != "JPEG" or source.width * source.height > 16_000_000:
                            raise ValueError("invalid JPEG")
                        decoded.append(source.convert("RGB"))
                return encoder.prepare(decoded)

            from starlette.concurrency import run_in_threadpool

            pixels = await run_in_threadpool(prepare)
        except (KeyError, TypeError, ValueError, OSError, binascii.Error):
            raise HTTPException(400, "invalid recognition request") from None

        prepared = time.perf_counter()

        def infer(pairs, error, timings):
            nonlocal ppe, shared_error

            try:
                if error is not None:
                    raise error
                rows = []
                for question, pair in zip(questions, pairs, strict=True):
                    try:
                        if not _valid_pair(pair):
                            raise ValueError("invalid model scores")
                    except (TypeError, ValueError) as error:
                        if question["id"] != "scene/ppe_violation":
                            raise
                        from .videojev_service import _json_safe

                        rows.append(
                            {
                                "id": question["id"],
                                "task": "choice",
                                "pair_scores": _json_safe(pair),
                                "prompts": list(PROMPTS["ppe_violation"]),
                                "error": {
                                    "type": type(error).__name__,
                                    "detail": str(error),
                                },
                            }
                        )
                        continue
                    index = 0 if pair[0] > pair[1] else 1
                    rows.append(
                        {
                            "id": question["id"],
                            "task": "choice",
                            "answer_index": index,
                            "answer": question["options"][index],
                            "probabilities": [*pair, 0.0],
                        }
                    )
                response = {
                    "request_id": payload["request_id"],
                    "model": MODEL,
                    "raw": rows,
                    "timings": {
                        **timings,
                        "server_preprocess_seconds": prepared - received,
                    },
                }
                if "ppe_violation" in labels:
                    from .ppe import (
                        ClipPPE,
                        attach_observation,
                        failed_observation,
                        shared_failure,
                    )

                    started = time.perf_counter()
                    try:
                        if shared_error is not None:
                            raise HTTPException(503, "recognition service CUDA unavailable")
                        if ppe is None:
                            ppe = ClipPPE(encoder)
                        sources = []
                        for encoded in images:
                            try:
                                sources.append(base64.b64decode(encoded, validate=True))
                            except (ValueError, TypeError, binascii.Error) as error:
                                sources.append(error)
                        observation = ppe.assess(sources, payload.get("frame_times"))
                    except Exception as error:
                        observation = getattr(
                            error, "ppe_observation", failed_observation(error)
                        )
                        if shared_failure(error):
                            shared_error = (
                                observation.get("error")
                                or failed_observation(error)["error"]
                            )
                    attach_observation(rows, observation)
                    response["timings"]["ppe_seconds"] = time.perf_counter() - started
                    if shared_error is not None:
                        response.update(
                            error="cuda_shared_failure", detail=shared_error["detail"]
                        )
                        return JSONResponse(status_code=503, content=response)
                response["timings"]["server_elapsed_seconds"] = time.perf_counter() - received
                return response
            except (TypeError, ValueError):
                raise HTTPException(503, "recognition could not complete") from None
            except Exception as error:
                from .ppe import failed_observation, shared_failure

                # The base classifier uses the same CUDA context as PPE. A
                # poisoned context must close admission regardless of which
                # forward encountered it first.
                if shared_failure(error):
                    shared_error = failed_observation(error)["error"]
                    return JSONResponse(
                        status_code=503,
                        content={
                            "request_id": payload["request_id"],
                            "error": "cuda_shared_failure",
                            "detail": shared_error["detail"],
                        },
                    )
                raise
        if shared_error is not None:
            raise HTTPException(503, "recognition service CUDA unavailable")
        if await request.is_disconnected():
            return JSONResponse(status_code=499, content={"detail": "client disconnected"})
        pending = asyncio.wrap_future(worker.submit(pixels, labels, infer))
        try:
            while not pending.done():
                await asyncio.wait({pending}, timeout=0.1)
                if not pending.done() and await request.is_disconnected():
                    pending.cancel()
                    return JSONResponse(status_code=499, content={"detail": "client disconnected"})
            return await pending
        except asyncio.CancelledError:
            pending.cancel()
            raise

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-batch-size", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    import uvicorn

    encoder = Encoder(args.cache_dir)
    encoder.warmup()
    uvicorn.run(create_app(encoder, max_batch_size=args.max_batch_size), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
