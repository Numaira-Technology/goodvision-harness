"""Reproducible paid VLM latency probe; credentials and images stay out of logs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from goodvision_harness.model_api import ModelAPISettings
from goodvision_harness.streaming_model_api import complete_stream
from goodvision_harness.video_to_stream import encode_jpeg
from goodvision_harness.vlm_rich_labeling import (
    build_compact_rich_annotation_messages,
    build_compact_rich_annotation_response_format,
    build_rich_annotation_messages,
    parse_rich_annotation,
)

DENSE_KEYS = ("a", "g", "d", "s", "uc", "lc", "b", "us", "ls", "gl", "ht", "bt", "ho")
DENSE_PROMPT = """Annotate all visible people AND prominent objects in this image.
Return minified JSON {"p":[person rows],"o":[object rows]}. No other text.
Each person row is exactly 17 values in this order:
[x1,y1,x2,y2,age,gender,direction,sleeves,upper_color,lower_color,bag,upper_pattern,lower_styles,glasses,hat,boots,holding_object].
Every field is required. Use JSON null for ANY uncertain or hidden attribute.
Coordinates are integers on the 0..1000 relative xyxy grid; tight VISIBLE boxes.
age: y=under18,a=18to60,o=over60; gender presentation: f=feminine,m=masculine;
direction: f=front,s=side,b=back; sleeves: s=short,l=long;
colors: short English color names; bag: n=none,h=handbag,s=shoulderbag,b=backpack;
upper_pattern: st=stripes,lo=logo,pl=plaid,sp=splice, or null when plain/unknown;
lower_styles: array with
tr=trousers,sh=shorts,sk=skirt/dress,co=longcoat,st=stripes,pa=pattern; or null if
hidden.
glasses,hat,boots,holding_object: 1=yes,0=no,null=uncertain/hidden.
Do not infer unseen pants/shoes or faces behind the head. Age/gender are broad
appearance estimates only.
Example person row (structure only):
[10,20,100,400,"a","m","s","s","white",null,"n",null,null,null,0,null,0].
Each object row: [label,x1,y1,x2,y2,color]. Short lowercase labels, color can be
null.
At most 12 rows combined. Include every visible person first, then up to 8 large
objects across categories (chairs, desks, monitors etc), with a separate tight box
for each instance. Never output only people when major objects are visible. Omit
tiny clutter, floor, walls and worn clothing. Clothing on a chair is not a person.
Empty scene: {"p":[],"o":[]}.
"""


def probe_schema(style: str) -> dict:
    if style == "packed":
        from vlm_packed_probe import response_format

        return response_format()
    result = build_compact_rich_annotation_response_format()
    row_variants = result["json_schema"]["schema"]["properties"]["o"]["items"]["anyOf"]
    person, item = row_variants
    attributes = person["prefixItems"][-1]
    attributes["required"] = list(DENSE_KEYS)
    attributes["type"] = "object"
    if style != "dense":
        return result
    coordinate = {"type": "integer", "minimum": 0, "maximum": 1000}
    person_row = {
        "type": "array",
        "minItems": 17,
        "maxItems": 17,
        "items": False,
        "prefixItems": [
            *[coordinate for _ in range(4)],
            *[attributes["properties"][k] for k in DENSE_KEYS],
        ],
    }
    item["minItems"] = item["maxItems"] = 6
    item["prefixItems"][-1] = attributes["properties"]["c"]
    result["json_schema"]["schema"] = {
        "type": "object",
        "required": ["p", "o"],
        "additionalProperties": False,
        "properties": {
            "p": {"type": "array", "maxItems": 12, "items": person_row},
            "o": {"type": "array", "maxItems": 8, "items": item},
        },
    }
    return result


def parse_probe(text: str, style: str):
    if style == "packed":
        from vlm_packed_probe import parse_result

        return parse_result(text)
    if style != "dense":
        return parse_rich_annotation(text)
    data = json.loads(text)
    if set(data) != {"p", "o"}:
        raise ValueError("dense response requires p and o")
    rows = []
    for person in data["p"]:
        if not isinstance(person, list) or len(person) != 17:
            raise ValueError("dense person must have 17 fields")
        rows.append(["p", *person[:4], dict(zip(DENSE_KEYS, person[4:], strict=True))])
    for item in data["o"]:
        if not isinstance(item, list) or len(item) != 6:
            raise ValueError("dense item must have 6 fields")
        rows.append([*item[:5], {"c": item[5]}])
    return parse_rich_annotation(json.dumps({"o": rows}))


def messages_for(image: bytes, style: str, cache: bool) -> list[dict]:
    if style == "packed":
        from vlm_packed_probe import build_messages

        return build_messages(image)
    messages = (
        build_rich_annotation_messages(image)
        if style == "objects"
        else build_compact_rich_annotation_messages(image)
    )
    if style == "complete":
        text = messages[1]["content"][0]["text"]
        text = text.replace(
            "Keep the whole response brief, about 300-400 tokens when possible.", ""
        )
        text += (
            " Every person MUST have all 13 keys a,g,d,s,uc,lc,b,us,ls,gl,ht,bt,ho; "
            "use null for uncertain or occluded values. Cover major objects of "
            "different categories, not only chairs. Minify the JSON."
        )
        messages[1]["content"][0]["text"] = text
    elif style == "dense":
        messages[1]["content"][0]["text"] = DENSE_PROMPT
    if cache:
        messages[1]["content"][0]["cache_control"] = {"type": "ephemeral"}
    return messages


async def run(args: argparse.Namespace) -> None:
    load_dotenv(args.env_file, override=False)
    key_name = (
        "OPENROUTER_API_KEY" if args.provider == "openrouter" else "DASHSCOPE_API_KEY"
    )
    key = os.environ.get(key_name, "").strip()
    if not key:
        raise SystemExit(f"{key_name} is missing")
    image = cv2.imdecode(
        np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise SystemExit("Image cannot be decoded")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    settings = ModelAPISettings(
        provider=args.provider,
        endpoint=args.endpoint or "",
        timeout_seconds=args.timeout,
        max_retries=0,
        extra_headers={"X-OpenRouter-Cache": "false"},
    )
    async with httpx.AsyncClient(follow_redirects=False) as client:
        if args.provider == "openrouter":
            try:
                auth = await client.get(
                    "https://openrouter.ai/api/v1/key",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=8,
                )
            except httpx.HTTPError:
                auth = None
            if auth is not None and auth.status_code in (401, 403):
                raise SystemExit(
                    f"OpenRouter key preflight failed (HTTP {auth.status_code}); "
                    "update the configured key"
                )
        try:
            trace = await client.get("https://openrouter.ai/cdn-cgi/trace", timeout=5)
            location = dict(
                line.split("=", 1) for line in trace.text.splitlines() if "=" in line
            )
        except httpx.HTTPError:
            location = {}
        print(
            json.dumps(
                {"exit_country": location.get("loc"), "edge": location.get("colo")}
            ),
            flush=True,
        )
        for trial in range(args.repeats):
            for case in args.case:
                model, provider, style = case.split(",")
                frame = cv2.flip(image, 1) if args.alternate and trial % 2 else image
                encoded, width, height = encode_jpeg(
                    frame, max_image_side=args.side, quality=85
                )
                messages = messages_for(encoded, style, args.cache)
                messages[1]["content"][0]["text"] *= args.prefix_repeats
                if args.cache_system:
                    instructions, image_block = messages[1]["content"]
                    instructions["text"] = (
                        messages[0]["content"] + "\n" + instructions["text"]
                    )
                    instructions["cache_control"] = {"type": "ephemeral"}
                    messages = [
                        {"role": "system", "content": [instructions]},
                        {"role": "user", "content": [image_block]},
                    ]
                elif args.cache_message:
                    instructions, image_block = messages[1]["content"]
                    instructions["cache_control"] = {"type": "ephemeral"}
                    messages = [
                        messages[0],
                        {"role": "user", "content": [instructions]},
                        {"role": "assistant", "content": "Ready."},
                        {"role": "user", "content": [image_block]},
                    ]
                params = {
                    "max_tokens": args.max_tokens,
                    "temperature": 0,
                    "reasoning": {"enabled": False},
                    "provider": {"only": [provider], "allow_fallbacks": False},
                }
                if args.provider == "dashscope":
                    params.pop("provider")
                    params.pop("reasoning")
                    params["enable_thinking"] = False
                if args.json:
                    params["response_format"] = {"type": "json_object"}
                if args.schema:
                    params["response_format"] = probe_schema(style)
                row = {
                    "case": case,
                    "trial": trial,
                    "cache_control": args.cache,
                    "prefix_repeats": args.prefix_repeats,
                    "cache_system": args.cache_system,
                    "cache_message": args.cache_message,
                    "image_sha256": hashlib.sha256(encoded).hexdigest(),
                    "dimensions": [width, height],
                    "image_bytes": len(encoded),
                }
                start = time.perf_counter()
                try:
                    response = await complete_stream(
                        client,
                        settings=settings,
                        model=model,
                        messages=messages,
                        api_key=key,
                        deadline_seconds=args.timeout,
                        request_parameters=params,
                    )
                    row.update(
                        {
                            "wall_seconds": round(time.perf_counter() - start, 6),
                            "response": response,
                        }
                    )
                    try:
                        objects = parse_probe(
                            response["choices"][0]["message"]["content"], style
                        )
                        people = [o for o in objects if o.label == "person"]
                        row.update(
                            {
                                "valid": True,
                                "people": len(people),
                                "items": len(objects) - len(people),
                                "labels": sorted({o.label for o in objects}),
                                "person_attribute_counts": [
                                    len(o.attributes) - 1 for o in people
                                ],
                            }
                        )
                    except Exception as exc:
                        row.update({"valid": False, "parse_error": str(exc)})
                except Exception as exc:
                    row.update(
                        {
                            "wall_seconds": round(time.perf_counter() - start, 6),
                            "valid": False,
                            "error": str(exc),
                        }
                    )
                with args.output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                brief = {
                    k: v
                    for k, v in row.items()
                    if k not in ("response", "image_sha256")
                }
                if "response" in row:
                    brief.update(
                        {
                            "usage": row["response"].get("usage"),
                            "timing": row["response"].get("_timing"),
                        }
                    )
                print(json.dumps(brief, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="model,provider,objects|compact|complete",
    )
    parser.add_argument(
        "--provider", choices=["openrouter", "dashscope"], default="openrouter"
    )
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--side", type=int, default=720)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--cache-system", action="store_true")
    parser.add_argument("--cache-message", action="store_true")
    parser.add_argument(
        "--prefix-repeats",
        type=int,
        default=1,
        help="Diagnostic only: repeat fixed text to probe cache threshold",
    )
    parser.add_argument(
        "--alternate",
        action="store_true",
        help="Alternate original/mirrored pixels to test new frames",
    )
    asyncio.run(run(parser.parse_args()))
