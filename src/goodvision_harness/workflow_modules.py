"""Stable canvas engine IDs, independent of vendor/model names."""

import os
from urllib.parse import urlsplit

ENGINES = {
    "tracking": ("contour", "bbox"),
    "events": ("vlm", "system1", "decision3", "neural"),
    # Object presence reuses the native VideoJev protocol; Decision 3 answers the
    # same questions on its own worker (experimental, not validated).
    "objects": ("system1", "decision3"),
}
NAMES = {
    "contour": "Contour",
    "bbox": "Bbox",
    "vlm": "VLM semantic recognition",
    "system1": "System 1 recognition",
    "decision3": "Decision 3 recognition",
    "neural": "Neural network recognition",
}
MODELS = {
    "system1": "ZhuYuhan/VideoJev-4B",
    "decision3": "vllm-sr/d3-nano",
    "neural": "openai/clip-vit-large-patch14",
}
# Engines that run in the native independent event lane (not the shared VLM).
NATIVE_EVENT_ENGINES = ("system1", "decision3", "neural")
NEURAL_EVENTS = ("fight", "smoking", "fall", "weapon", "ppe_violation", "ground_litter")
POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")
LOCAL_PORTS = {"bbox": 8096, "system1": 8097, "neural": 8098, "decision3": 8099}
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
    if not isinstance(value, dict) or set(value) - {
        "tracks", "events", "summary", "objects",
    }:
        raise ValueError("invalid output display settings")
    result = {}
    for kind, default in (
        ("tracks", "top-left"),
        ("events", "top-left"),
        ("summary", "top-right"),
        ("objects", "bottom-left"),
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
        # Validate before persisting or turning a local URL into launch arguments.
        # Port zero is for binding an ephemeral server, never a client endpoint.
        if parsed.port == 0:
            raise ValueError("Service URL port must be within 1..65535.")
    return value


# Module contract: typed ports per node kind. ``frames`` stays the required
# input of every analysis node; any upstream input added later is optional and
# lists the (kind, engine) modules it accepts. Old graphs never use an optional
# input, so their validation and compiled plans are unchanged.
PORT_TYPES = ("frames", "tracks", "faces", "events", "summary", "objects")
CONTRACT = {
    "source": {"inputs": (), "outputs": ("frames",)},
    "face": {"inputs": (("frames", True),), "outputs": ("faces",)},
    "tracking": {"inputs": (("frames", True),), "outputs": ("tracks",)},
    "events": {"inputs": (("frames", True),), "outputs": ("events",)},
    "summary": {"inputs": (("frames", True),), "outputs": ("summary",)},
    # Bbox person boxes focus VideoJev on people (person crop). Contour tracks
    # carry no boxes, so only Bbox is accepted.
    "objects": {
        "inputs": (("frames", True), ("tracks", False)),
        "outputs": ("objects",),
    },
    "output": {
        "inputs": tuple((port, False) for port in PORT_TYPES),
        "outputs": (),
    },
}
ACCEPTS = {("objects", "tracks"): (("tracking", "bbox"),)}
# Task modules (object presence, summary) ask a task of a recognizer chosen in
# the node. Every recognizer is its own peer module with its own block, service,
# health check and debug entry; System 1 and Decision 3 only share transport code.
TASK_KINDS = ("objects", "summary")
# How each peer module reports, so a later decision node can consume them alike.
OUTPUT_CONTRACTS = {
    "contour": "person_contours",
    "bbox": "person_boxes",
    "system1": "native_option_probabilities",
    "decision3": "native_option_probabilities",
    "neural": "embedding_margin",
    "vlm": "generative_json",
}
LAYERS = {"tracking": "perception", "face": "perception", "events": "understanding"}
# Nodes whose debug entry has a per-request record, not only a status line.
DEBUG_RECORDS = {"objects", "summary", "tracking", "events"}


def node_ports(kind):
    """{"inputs": (port, ...), "outputs": (port, ...)} for validation."""
    spec = CONTRACT[kind]
    return {
        "inputs": tuple(port for port, _ in spec["inputs"]),
        "outputs": spec["outputs"],
    }


def required_inputs(kind):
    return tuple(port for port, required in CONTRACT[kind]["inputs"] if required)


def accepts(kind, port, source_kind, source_engine):
    allowed = ACCEPTS.get((kind, port))
    return allowed is None or (source_kind, source_engine) in allowed


def module_catalog():
    """JSON contract the canvas reads instead of keeping its own port table."""
    return {
        "port_types": list(PORT_TYPES),
        "modules": {
            kind: {
                "inputs": [
                    {
                        "port": port,
                        "type": port,
                        "required": required,
                        **(
                            {"accepts": [list(a) for a in ACCEPTS[(kind, port)]]}
                            if (kind, port) in ACCEPTS
                            else {}
                        ),
                    }
                    for port, required in spec["inputs"]
                ],
                "outputs": [{"port": port, "type": port} for port in spec["outputs"]],
                "engines": [
                    {
                        "id": engine,
                        "name": NAMES[engine],
                        "model": MODELS.get(engine),
                        "local": engine in LOCAL_PORTS,
                    }
                    for engine in ENGINES.get(kind, ())
                ],
                "debug": "records" if kind in DEBUG_RECORDS else "status",
                "task": kind in TASK_KINDS,
            }
            for kind, spec in CONTRACT.items()
        },
        "peers": [
            {
                "id": engine,
                "kind": kind,
                "layer": LAYERS[kind],
                "name": NAMES[engine],
                "model": MODELS.get(engine),
                "local_port": LOCAL_PORTS.get(engine),
                "output_contract": OUTPUT_CONTRACTS[engine],
            }
            for kind in ("tracking", "events")
            for engine in ENGINES[kind]
        ],
    }
