"""Standalone CUDA classifier. Install transformers/torch separately to serve it."""

import argparse
import base64
import binascii
import io
import json
import math
import threading

from fastapi import FastAPI, HTTPException, Request
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
        "A visible person is missing a safety helmet or a reflective safety vest, or both.",
        "All visible people are wearing both safety helmets and reflective safety vests.",
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


def create_app(encoder):
    app, lock = FastAPI(), threading.Lock()

    @app.get("/health")
    def health():
        return {"ready": True, "model": MODEL}

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
            # Own the lock in the worker: disconnect/cancellation must not admit
            # another GPU operation while this one is still running.
            if not lock.acquire(blocking=False):
                raise HTTPException(429, "recognition service busy")
            try:
                pairs = encoder.scores(image, labels)
                rows = []
                for question, pair in zip(questions, pairs, strict=True):
                    if (
                        len(pair) != 2
                        or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pair)
                        or abs(sum(pair) - 1) > 1e-5
                    ):
                        raise ValueError("invalid model scores")
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
                return {
                    "request_id": payload["request_id"],
                    "model": MODEL,
                    "raw": rows,
                    "timings": {},
                }
            except (TypeError, ValueError):
                raise HTTPException(503, "recognition could not complete") from None
            finally:
                lock.release()

        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(infer)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(Encoder(args.cache_dir)), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
