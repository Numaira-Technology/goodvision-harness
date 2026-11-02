"""Stable canvas engine IDs, independent of vendor/model names."""

import os
from urllib.parse import urlsplit

ENGINES = {"tracking": ("contour", "bbox"), "events": ("vlm", "system1", "neural")}
NAMES = {
    "contour": "Contour",
    "bbox": "Bbox",
    "vlm": "VLM semantic recognition",
    "system1": "System 1 recognition",
    "neural": "Neural network recognition",
}
MODELS = {"system1": "ZhuYuhan/VideoJev-4B", "neural": "openai/clip-vit-large-patch14"}
NEURAL_EVENTS = ("fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter")
POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")
LOCAL_PORTS = {"bbox": 8096, "system1": 8097, "neural": 8098}
EXECUTION_MODES = ("local", "server", "cloud")


def execution_mode(config):
    # Existing workflows with an address continue to use their deployed worker.
    return config.get(
        "execution_mode", "server" if config.get("service_url") else "local"
    )


def module_endpoint(engine, config):
    mode = execution_mode(config)
    if mode == "local":
        endpoint = os.environ.get(
            f"GOODVISION_{engine.upper()}_LOCAL_URL",
            f"http://127.0.0.1:{LOCAL_PORTS[engine]}",
        )
    else:
        endpoint = config.get(
            "cloud_service_url" if mode == "cloud" else "service_url", ""
        )
    endpoint = service_url(endpoint).rstrip("/")
    if engine == "bbox" and endpoint and not urlsplit(endpoint).path:
        endpoint += "/v1/labels"
    return endpoint


def display_settings(value):
    if not isinstance(value, dict) or set(value) - {"tracks", "events", "summary"}:
        raise ValueError("invalid output display settings")
    result = {}
    for kind, default in (
        ("tracks", "top-left"),
        ("events", "top-left"),
        ("summary", "top-right"),
    ):
        item = value.get(kind, {})
        if not isinstance(item, dict) or set(item) - {"visible", "position"}:
            raise ValueError("invalid output layer")
        visible, position = item.get("visible", True), item.get("position", default)
        if type(visible) is not bool or position not in POSITIONS:
            raise ValueError("invalid output layer visibility or position")
        result[kind] = {"visible": visible, "position": position}
    return result


def engine_for(kind, config):
    return config.get("engine", ENGINES[kind][0]) if kind in ENGINES else kind


def native_endpoints(endpoint):
    base = endpoint.rstrip("/").removesuffix("/predict")
    return base + "/predict", base + "/health"


def service_url(value):
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("Service URL must be an HTTP(S) address without credentials.")
    value = value.strip()
    parsed = urlsplit(value)
    if value and (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or any(ord(c) <= 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("Service URL must be an HTTP(S) address without credentials.")
    if value:
        _ = parsed.port  # Validate malformed or out-of-range ports before persisting.
    return value
