"""仅正式部署下载固定官方权重，保存字节、摘要和来源，不加载模型。"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

SOURCES = {
    "yolo26-s": {
        "filename": "yolo26s.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
        "upstream": "ultralytics/assets release v8.4.0",
    },
    "rfdetr-l": {
        "filename": "rf-detr-large.pth",
        "url": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
        "upstream": "roboflow/rf-detr tag 1.3.0 rfdetr/main.py HOSTED_MODELS",
    },
}

def main():
    directory = Path(sys.argv[1])
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"recorded_epoch": time.time(), "models": {}}
    for model, source in SOURCES.items():
        target = directory / source["filename"]
        if target.exists():
            raise RuntimeError("拒绝覆盖已有权重；需按保留证据的恢复流程处理")
        temporary = target.with_suffix(target.suffix + ".partial")
        digest = hashlib.sha256()
        size = 0
        request = Request(source["url"], headers={"User-Agent": "GoodVision-detector-evaluation"})
        with urlopen(request, timeout=120) as response, temporary.open("xb") as out:
            remote_metadata = {
                key: response.headers.get(key)
                for key in ("ETag", "Content-Length", "x-goog-generation", "x-goog-hash")
            }
            resolved_url = response.url.split("?")[0]
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            out.flush()
            os.fsync(out.fileno())
        if not size:
            raise RuntimeError("官方权重下载为空")
        os.replace(temporary, target)
        manifest["models"][model] = {
            **source, "bytes": size, "sha256": digest.hexdigest(),
            "resolved_url_without_query": resolved_url,
            "response_metadata": remote_metadata,
        }
        temporary_manifest = directory / "weights.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary_manifest, directory / "weights.json")
        print("DETECTOR_WEIGHT " + json.dumps(manifest["models"][model], ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()

