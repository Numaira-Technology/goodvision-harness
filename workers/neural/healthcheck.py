"""Only a warmed, healthy CLIP service can satisfy the container health check."""

import json
import urllib.request


def main():
    with urllib.request.urlopen("http://127.0.0.1:8098/health", timeout=3) as response:
        status = json.load(response)
    if (status.get("ready") is not True
            or status.get("model") != "openai/clip-vit-large-patch14"):
        raise SystemExit("CLIP is not ready")


if __name__ == "__main__":
    main()
