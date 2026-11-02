"""Standalone CUDA classifier. Install transformers/torch separately to serve it."""

import argparse
import base64
import binascii
import io
import json
import math
import threading
import time

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

    def warmup(self):
        # Loading weights alone does not prove compatible CUDA kernels or the
        # processor/model interface. Do not publish ready until inference works.
        labels = list(PROMPTS)
        pairs = self.scores(Image.new("RGB", (224, 224)), labels)
        if len(pairs) != len(labels) or not all(_valid_pair(pair) for pair in pairs):
            raise ValueError("invalid model scores during warmup")

    def scores(self, image, labels):
        torch = self.torch
        inputs = self.processor(
            images=image,
            text=[text for label in labels for text in PROMPTS[label]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.inference_mode():
            result = self.model(
                **{key: value.to("cuda") for key, value in inputs.items()}
            )
        return (
            result.logits_per_image[0]
            .reshape(len(labels), 2)
            .softmax(dim=-1)
            .cpu()
            .tolist()
        )


def _valid_pair(pair):
    return (
        len(pair) == 2
        and all(math.isfinite(p) and 0 <= p <= 1 for p in pair)
        and abs(sum(pair) - 1) <= 1e-5
    )


def create_app(encoder):
    app, lock = FastAPI(), threading.Lock()
    ppe, shared_error = None, None

    @app.get("/health")
    def health():
        return {"ready": shared_error is None, "model": MODEL}

    @app.post("/predict")
    async def predict(request: Request):
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
            with Image.open(
                io.BytesIO(base64.b64decode(images[-1], validate=True))
            ) as source:
                if source.format != "JPEG" or source.width * source.height > 16_000_000:
                    raise ValueError("invalid JPEG")
                image = source.convert("RGB")
        except (KeyError, TypeError, ValueError, OSError, binascii.Error):
            raise HTTPException(400, "invalid recognition request") from None

        def infer():
            nonlocal ppe, shared_error

            # Own the lock in the worker: disconnect/cancellation must not admit
            # another GPU operation while this one is still running.
            if not lock.acquire(blocking=False):
                raise HTTPException(429, "recognition service busy")
            try:
                if shared_error is not None:
                    raise HTTPException(503, "recognition service CUDA unavailable")
                pairs = encoder.scores(image, labels)
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
                    "timings": {},
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
            finally:
                lock.release()

        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(infer)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    import uvicorn

    encoder = Encoder(args.cache_dir)
    encoder.warmup()
    uvicorn.run(create_app(encoder), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
