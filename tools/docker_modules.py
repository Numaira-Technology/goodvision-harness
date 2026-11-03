"""Standard-library-only deployment definitions shared by installers and exports."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

MODULES = ("main", "contour", "bbox", "system1", "gpu-runtime")
PORTS = {"main": 8080, "contour": 8093, "bbox": 8096, "system1": 8097}
TORCH = {"2.10": ("2.10.0", "0.25.0"), "2.8": ("2.8.0", "0.23.0")}
IMAGE_ROLES = {"main": {"main"}, "bbox": {"bbox"},
               "contour": {"contour", "engine"}, "system1": {"system1"},
               "gpu-runtime": {"torch2.10", "torch2.8"}}


# Every writer of one deployment shares this lock, including configure/build/up.
# Keep the file after unlocking: unlinking it can split waiters across inodes.
_STATE_LOCKS = {}
_STATE_LOCKS_GUARD = threading.Lock()
_STATE_HELD = threading.local()


@contextmanager
def state_lock(state):
    state = state.resolve()
    with _STATE_LOCKS_GUARD:
        lock = _STATE_LOCKS.setdefault(state, threading.RLock())
    with lock:
        held = getattr(_STATE_HELD, "paths", set())
        if state in held:
            yield False
            return
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (state / ".installer.lock").open("a+b") as stream:
            if os.name == "nt":
                import msvcrt
                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                deadline = time.monotonic() + 900
                while True:
                    stream.seek(0)
                    try:
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise ValueError(
                                "Deployment is busy; retry later"
                            ) from None
                        time.sleep(.1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            _STATE_HELD.paths = held | {state}
            try:
                yield True
            finally:
                _STATE_HELD.paths = held
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def bbox_health(device):
    capability = (
        "r['device']=='cpu' and r['runtime']=='ultralytics' and "
        "r['provider_ready'] is True and r['gpu_ready'] is False"
        if device == "cpu" else "r['gpu_ready'] is True"
    )
    return health(
        "import json,urllib.request; r=json.load(urllib.request.urlopen("
        "'http://127.0.0.1:8096/health/tracking',timeout=3)); "
        "assert r['ready'] is True and r['model']=='yolo26-s' and " + capability
    )


def image_role(module, image):
    ref = image["name"]
    if not re.fullmatch(
        rf"goodvision-{module}:[a-zA-Z0-9_][a-zA-Z0-9_.-]{{0,127}}", ref,
    ):
        raise ValueError("Image belongs to another module or has an invalid tag")
    if module == "contour":
        for suffix, role in (("-engine", "engine"), ("-api", "contour")):
            if ref.endswith(suffix):
                return role
        raise ValueError("Contour image must use an explicit -api or -engine role")
    if module == "gpu-runtime":
        for version in TORCH:
            if ref.endswith(f"-torch{version}-cu128"):
                return "torch" + version
        raise ValueError("Unsupported GPU runtime image role")
    return module


def name(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]{0,39}", value
    ):
        raise ValueError("Use 1-40 lowercase letters, digits, underscores or hyphens.")
    return value


def endpoint(value):
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("Invalid service address.")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or any(ord(c) <= 32 or ord(c) == 127 for c in value)
        or parsed.port == 0
    ):
        raise ValueError("Use an HTTP(S) service address without credentials or query.")
    return value.rstrip("/")


def defaults(namespace="default", version="dev"):
    return {
        "namespace": name(namespace),
        "version": name(version),
        "module_versions": {module: name(version) for module in MODULES},
        "bbox_device": "cpu",
        "bind": "127.0.0.1",
        "gpu_id": "0",
        "contour_capacity": 1,
        "ports": dict(PORTS),
        "urls": {
            "bbox": "http://gv-bbox:8096",
            "system1": "http://gv-system1:8097",
            "neural": "http://gv-neural:8098",
            "contour": "http://gv-contour:8093",
        },
    }


def validate(settings):
    name(settings["namespace"])
    name(settings["version"])
    versions = settings.get("module_versions", {})
    if not isinstance(versions, dict) or set(versions) - set(MODULES):
        raise ValueError("Invalid module versions")
    for value in versions.values():
        name(value)
    images = settings.get("module_images", {})
    if not isinstance(images, dict) or set(images) - set(MODULES):
        raise ValueError("Invalid module image references")
    for module, roles in images.items():
        if not isinstance(roles, dict) or set(roles) - IMAGE_ROLES[module]:
            raise ValueError("Invalid module image roles")
        for value in roles.values():
            if not isinstance(value, str) or not re.fullmatch(
                rf"goodvision-{module}:[a-zA-Z0-9_][a-zA-Z0-9_.-]{{0,127}}", value
            ):
                raise ValueError("Invalid module image reference")
    ipaddress.ip_address(settings["bind"])
    if settings["bbox_device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("bbox_device must be auto, cpu or cuda")
    if settings.get("bbox_runtime_device", "cpu") not in ("cpu", "cuda"):
        raise ValueError("Invalid resolved Bbox device")
    if settings.get("bbox_platform", "linux/amd64") not in (
        "linux/amd64", "linux/arm64",
    ):
        raise ValueError("Invalid Bbox platform")
    if not re.fullmatch(r"[0-9]+|GPU-[a-fA-F0-9-]+", settings["gpu_id"]):
        raise ValueError("Invalid NVIDIA GPU identifier")
    if (
        type(settings["contour_capacity"]) is not int
        or settings["contour_capacity"] < 1
    ):
        raise ValueError("Contour capacity must be a positive integer")
    if set(settings["ports"]) != set(PORTS) or any(
        type(port) is not int or not 1 <= port <= 65535
        for port in settings["ports"].values()
    ):
        raise ValueError("Ports must be within 1..65535")
    if len(set(settings["ports"].values())) != len(PORTS):
        raise ValueError("Module host ports must be different")
    if set(settings["urls"]) != {"bbox", "system1", "neural", "contour"}:
        raise ValueError("Missing module service addresses")
    for value in settings["urls"].values():
        endpoint(value)
    return settings


def bbox_device(settings):
    policy = settings["bbox_device"]
    return settings.get("bbox_runtime_device", "cpu") if policy == "auto" else policy


def bbox_image(settings):
    pinned = settings.get("module_images", {}).get("bbox", {}).get("bbox")
    if pinned:
        return pinned
    version = settings.get("module_versions", {}).get("bbox", settings["version"])
    suffix = (
        "-" + settings["bbox_platform"].split("/")[1]
        if "bbox_platform" in settings else ""
    )
    return f"goodvision-bbox:{version}-{bbox_device(settings)}{suffix}"


def runtime_image(settings, torch):
    runtimes = settings.get("module_images", {}).get("gpu-runtime", {})
    pinned = runtimes.get("torch" + torch)
    if pinned:
        return pinned
    versions = settings.get("module_versions", {})
    version = versions.get("gpu-runtime", settings["version"])
    return f"goodvision-gpu-runtime:{version}-torch{torch}-cu128"


def runtime_versions(modules, settings):
    versions = set()
    if "contour" in modules or (
        "bbox" in modules and bbox_device(settings) == "cuda"
    ):
        versions.add("2.10")
    if "system1" in modules:
        versions.add("2.8")
    if "gpu-runtime" in modules:
        versions.update(TORCH)
    return sorted(versions)


def runtime_build(source, settings, torch):
    version, vision = TORCH[torch]
    return [
        "docker",
        "build",
        "--platform",
        "linux/amd64",
        "-f",
        str(source / "workers/gpu_runtime/Dockerfile"),
        "--build-arg",
        f"TORCH_VERSION={version}",
        "--build-arg",
        f"TORCHVISION_VERSION={vision}",
        "-t",
        runtime_image(settings, torch),
        str(source),
    ]


def health(script, *, start="120s"):
    return {
        "test": ["CMD", "python", "-c", script],
        "interval": "15s",
        "timeout": "5s",
        "start_period": start,
        "retries": 4,
    }


def mount(source, target, *, readonly=False):
    return {
        "type": "bind",
        "source": str(source),
        "target": target,
        "read_only": readonly,
        "bind": {"create_host_path": False},
    }


def compose(module, settings, source: Path, state: Path):
    """One independently managed project; only the named network is shared."""
    validate(settings)
    if module not in PORTS:
        raise ValueError("The GPU runtime is a base image, not a running service")
    ns = settings["namespace"]
    version = settings.get("module_versions", {}).get(module, settings["version"])
    project = f"goodvision-{module}-{ns}"
    bind = settings["bind"]
    if ":" in bind:
        bind = f"[{bind}]"
    service = {
        "container_name": f"goodvision-{module}-{ns}",
        "init": True,
        "restart": "unless-stopped",
        "ports": [f"{bind}:{settings['ports'][module]}:{PORTS[module]}"],
        "networks": {"modules": {"aliases": [f"gv-{module}"]}},
        "logging": {"options": {"max-size": "10m", "max-file": "3"}},
    }
    document = {
        "name": project,
        "services": {module: service},
        "networks": {"modules": {"external": True, "name": f"goodvision-{ns}"}},
        "volumes": {"data": {}},
    }
    service["volumes"] = ["data:/data"]
    if module == "main":
        service.update(
            image=f"goodvision-main:{version}",
            build={"context": str(source), "dockerfile": "Dockerfile"},
        )
        service["environment"] = {
            "HARNESS_HOST": "0.0.0.0",
            "HARNESS_PORT": "8080",
            "HARNESS_CONFIG": "/app/config.yaml",
            "HARNESS_AUTOSTART": "false",
            "HARNESS_DRY_RUN": "false",
            "HARNESS_STREAM_URL": "",
            "HARNESS_DEPLOYMENT_SETTINGS_PATH": "/data/deployment-settings.json",
            "EVENT_EVIDENCE_DIRECTORY": "/data/events",
            "FACE_MODEL_ROOT": "/models",
            "FACE_GALLERY_PATH": "/data/face-gallery.json",
            "GOODVISION_MODULE_MANAGER": "compose",
            "GOODVISION_MODULE_DATA_DIR": "/data/modules",
            **{
                f"GOODVISION_{key.upper()}_LOCAL_URL": settings["urls"][key]
                for key in ("bbox", "system1", "neural")
            },
            "TRACKING_EXECUTION_MODE": "server",
            "TRACKING_SERVER_PROTOCOL": "contour",
            "TRACKING_SERVER_URL": settings["urls"]["contour"],
            "TRACKING_SERVER_CAPACITY": str(settings["contour_capacity"]),
        }
        document["volumes"]["face-models"] = {}
        service["volumes"].append("face-models:/models")
        service["extra_hosts"] = ["host.docker.internal:host-gateway"]
        service["healthcheck"] = health(
            "import urllib.request; urllib.request.urlopen("
            "'http://127.0.0.1:8080/health/live',timeout=3).read()"
        )
    elif module == "bbox":
        device = bbox_device(settings)
        service.update(
            image=bbox_image(settings),
            build={
                "context": str(source),
                "dockerfile": "workers/bbox/Dockerfile.cpu"
                if device == "cpu"
                else "workers/bbox/Dockerfile",
                "args": {"GPU_RUNTIME_IMAGE": runtime_image(settings, "2.10")},
            },
        )
        if "bbox_platform" in settings:
            service["platform"] = settings["bbox_platform"]
        service["environment"] = {
            "DETECTOR_DEVICE": device,
            "DETECTOR_MODEL": "yolo26-s",
            "DETECTOR_WEIGHTS": "/models/yolo26s.pt",
        }
        service["volumes"].append(mount(state / "models/bbox", "/models"))
        service["healthcheck"] = bbox_health(device)
        if device == "cuda":
            service["deploy"] = gpu(settings)
    elif module == "system1":
        service.update(
            image=f"goodvision-system1:{version}",
            build={
                "context": str(source),
                "dockerfile": "workers/system1/Dockerfile",
                "args": {"GPU_RUNTIME_IMAGE": runtime_image(settings, "2.8")},
            },
            deploy=gpu(settings),
            shm_size="2g",
        )
        service["volumes"].append(mount(state / "models/system1", "/models"))
        service["healthcheck"] = health(
            "import json,urllib.request; r=json.load(urllib.request.urlopen("
            "'http://127.0.0.1:8097/health',timeout=3)); "
            "assert r['ready'] is True and r['model']=='ZhuYuhan/VideoJev-4B'",
            start="600s",
        )
    else:
        service.update(
            # Both roles need disjoint suffixes: a namespace/version ending in
            # "-engine" must not reuse another installation's engine identity.
            container_name=f"goodvision-contour-{ns}-api",
            image=f"goodvision-contour:{version}-api",
            build={
                "context": str(source / "workers/contour"),
                "dockerfile": "Dockerfile",
            },
        )
        service["environment"] = {
            "CONTOUR_API_KEY_FILE": "/run/secrets/node-key",
            "CONTOUR_ENGINE_URL": "http://engine:8092",
            "CONTOUR_CHECKPOINT_PATH": "/models/contour.pt",
            "CONTOUR_CAPACITY": str(settings["contour_capacity"]),
        }
        model_mount = mount(state / "models/contour", "/models", readonly=True)
        service["volumes"] = [
            model_mount,
            mount(
                state / "secrets/contour-key", "/run/secrets/node-key", readonly=True
            ),
        ]
        service["networks"]["engine"] = {}
        service["depends_on"] = {"engine": {"condition": "service_healthy"}}
        service["healthcheck"] = health(
            "import urllib.request; from pathlib import Path; "
            "r=urllib.request.Request('http://127.0.0.1:8093/health/ready',headers="
            "{'Authorization':'Bearer '+"
            "Path('/run/secrets/node-key').read_text().strip()}); "
            "urllib.request.urlopen(r,timeout=3).read()",
            start="600s",
        )
        document["networks"]["engine"] = {"internal": True}
        document["services"]["engine"] = {
            "container_name": f"goodvision-contour-{ns}-engine",
            "image": f"goodvision-contour:{version}-engine",
            "build": {
                "context": str(source / "workers"),
                "dockerfile": "contour/Dockerfile.engine",
                "args": {
                    "GPU_RUNTIME_IMAGE": runtime_image(settings, "2.10"),
                    "TORCH_CUDA": "cu128",
                },
            },
            "init": True,
            "restart": "unless-stopped",
            "networks": ["engine"],
            "environment": {
                "SAM_SCENE_IMAGE_ENABLED": "false",
                "SAM31_ENABLED": "true",
                "SAM31_SESSION_ENABLED": "false",
                "SAM31_LIVE_ENABLED": "true",
                "SAM31_LIVE_CAMERAS": str(settings["contour_capacity"]),
                "SAM31_SESSION_H264_ENABLED": "true",
                "SAM31_CHECKPOINT_PATH": "/models/contour.pt",
            },
            "volumes": [model_mount],
            "deploy": gpu(settings),
            "shm_size": "2g",
            "stop_grace_period": "90s",
            "logging": service["logging"],
            "healthcheck": health(
                "import json,urllib.request; r=json.load(urllib.request.urlopen("
                "'http://127.0.0.1:8092/health/perception/sam_live',timeout=3)); "
                "assert r['enabled'] and r['ready'] and r['gpu_ready']",
                start="600s",
            ),
        }
    for role, image in settings.get("module_images", {}).get(module, {}).items():
        document["services"][role]["image"] = image
    # A project-name migration pins the original physical resources. Keep those
    # bindings when configure/build/up subsequently regenerate the recipe.
    saved = state / f"compose.{module}.json"
    if saved.is_file():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        for section in ("volumes", "networks"):
            for key in document.get(section, {}):
                resource = previous.get(section, {}).get(key) or {}
                if resource.get("external") and resource.get("name"):
                    document[section][key] = resource
    return compose_literals(document)


def compose_literals(value):
    """User paths/URLs are literals, not Compose environment substitutions."""
    if isinstance(value, str):
        return value.replace("$", "$$")
    if isinstance(value, list):
        return [compose_literals(item) for item in value]
    if isinstance(value, dict):
        return {key: compose_literals(item) for key, item in value.items()}
    return value


def gpu(settings):
    return {
        "resources": {
            "reservations": {
                "devices": [
                    {
                        "driver": "nvidia",
                        "device_ids": [settings["gpu_id"]],
                        "capabilities": ["gpu"],
                    }
                ]
            }
        }
    }
