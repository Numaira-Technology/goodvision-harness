"""Install the independent CLIP container beside an existing GoodVision V2.0.

Standard library only; copy this file with RELEASE.json and the image archive.
The existing installer, settings, other containers and models are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

MODULE = "goodvision-neural"
MODEL = "openai/clip-vit-large-patch14"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
MODEL_FILES = {
    "config.json": 4519,
    "merges.txt": 524619,
    "model.safetensors": 1710540580,
    "preprocessor_config.json": 316,
    "special_tokens_map.json": 389,
    "tokenizer.json": 2224003,
    "tokenizer_config.json": 905,
    "vocab.json": 961143,
}
HEALTH = (
    "import json,urllib.request; r=json.load(urllib.request.urlopen("
    "'http://127.0.0.1:8098/health',timeout=3)); "
    "assert r['ready'] is True and r['model']=='" + MODEL + "'"
)
LABEL = "io.goodvision.neural.installation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def identity(entry, *, neural=False):
    prefix = "goodvision-neural" if neural else "goodvision-system1"
    if (
        not re.fullmatch(
            prefix + r":[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", entry.get("name", "")
        )
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", entry.get("id", ""))
        or entry.get("os") != "linux"
        or entry.get("architecture") != "amd64"
    ):
        raise ValueError("Invalid Linux/amd64 image identity in RELEASE.json")
    if "config_id" in entry and not re.fullmatch(
        r"sha256:[a-f0-9]{64}", entry["config_id"]
    ):
        raise ValueError("Invalid image config identity")


def release(directory):
    value = read(directory / "RELEASE.json")
    if value.get("module") != MODULE or not re.fullmatch(
        r"v[0-9]+\.[0-9]+(?:\.[0-9]+)?", value.get("version", "")
    ):
        raise ValueError("This is not an independent goodvision-neural release")
    identity(value["image"], neural=True)
    for entry in value.get("requires", []):
        identity(entry)
    archive = value["image_archive"]
    name = archive.get("file", "")
    if (
        not name
        or Path(name).name != name
        or "\\" in name
        or ":" in name
        or type(archive.get("bytes")) is not int
        or archive["bytes"] < 1
        or not re.fullmatch(r"[a-f0-9]{64}", archive.get("sha256", ""))
    ):
        raise ValueError("Invalid image archive metadata")
    return value


def environment(service):
    values = service.get("environment", {})
    if isinstance(values, list):
        return dict(item.split("=", 1) for item in values if "=" in item)
    if not isinstance(values, dict):
        raise ValueError("Invalid Main environment")
    return values


def deployment(goodvision_dir, info, host_port=28098):
    state = goodvision_dir.resolve() / "state"
    settings = read(state / "settings.json")
    main = read(state / "compose.main.json")
    candidates = [
        service
        for service in main["services"].values()
        if "GOODVISION_NEURAL_LOCAL_URL" in environment(service)
    ]
    if len(candidates) != 1:
        raise ValueError("Cannot identify Main's configured neural service URL")
    service = candidates[0]
    url = environment(service)["GOODVISION_NEURAL_LOCAL_URL"]
    if settings.get("urls", {}).get("neural", url) != url:
        raise ValueError(
            "Main and settings neural URLs differ; original files left unchanged"
        )
    parsed = urlsplit(url)
    alias = parsed.hostname or ""
    if (
        parsed.scheme != "http"
        or parsed.port != 8098
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,252}", alias)
    ):
        raise ValueError(
            "Main must address CLIP using an HTTP Docker alias on port 8098"
        )
    if (
        alias == "localhost"
        or alias.endswith(".localhost")
        or alias
        in (
            "host.docker.internal",
            "gateway.docker.internal",
        )
    ):
        raise ValueError(
            "Main's neural URL must use a service alias, not a host address"
        )
    try:
        ipaddress.ip_address(alias)
    except ValueError:
        pass
    else:
        raise ValueError("Main's neural URL must use a Docker network alias")
    networks = service.get("networks", {})
    external = [
        main["networks"][key]["name"]
        for key in networks
        if main.get("networks", {}).get(key, {}).get("external") is True
        and main["networks"][key].get("name")
    ]
    if len(external) != 1:
        raise ValueError(
            "Cannot uniquely identify Main's existing shared Docker network"
        )
    network = external[0]
    if not isinstance(network, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", network
    ):
        raise ValueError("Invalid shared Docker network name")
    gpu = settings.get("gpu_id")
    if not isinstance(gpu, str) or not re.fullmatch(r"[0-9]+|GPU-[a-fA-F0-9-]+", gpu):
        raise ValueError("Missing or invalid NVIDIA GPU identifier in settings.json")
    bind = settings.get("bind")
    ipaddress.ip_address(bind)
    if type(host_port) is not int or not 1 <= host_port <= 65535:
        raise ValueError("Host port must be within 1..65535")
    if host_port in settings.get("ports", {}).values():
        raise ValueError("CLIP host port overlaps an existing module port")
    models = state / "models" / "neural"
    snapshot = (
        models
        / "huggingface"
        / "models--openai--clip-vit-large-patch14"
        / "snapshots"
        / REVISION
    )
    for name, size in MODEL_FILES.items():
        path = snapshot / name
        if (
            not path.is_file()
            or path.stat().st_size != size
            or not path.resolve().is_relative_to(models.resolve())
        ):
            raise ValueError(
                "Missing or incomplete original CLIP model file: " + str(path)
            )
    person_model = models / "yolo26s.pt"
    if not person_model.is_file() or person_model.stat().st_size != 20422725:
        raise ValueError(
            "Missing or incomplete original PPE person model: " + str(person_model)
        )
    document = {
        "name": MODULE,
        "services": {
            "neural": {
                "image": info["image"]["name"],
                "platform": "linux/amd64",
                "container_name": MODULE,
                "init": True,
                "restart": "unless-stopped",
                "pull_policy": "never",
                "labels": {
                    LABEL: str(state),
                    "io.goodvision.module": MODULE,
                    "io.goodvision.version": info["version"],
                },
                "environment": {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HOME": "/tmp/huggingface",
                },
                "ports": [
                    {
                        "target": 8098,
                        "published": str(host_port),
                        "host_ip": bind,
                        "protocol": "tcp",
                    }
                ],
                "networks": {"modules": {"aliases": [alias]}},
                "volumes": [
                    {
                        "type": "bind",
                        "source": str(models),
                        "target": "/models",
                        "read_only": True,
                        "bind": {"create_host_path": False},
                    }
                ],
                "deploy": {
                    "resources": {
                        "reservations": {
                            "devices": [
                                {
                                    "driver": "nvidia",
                                    "device_ids": [gpu],
                                    "capabilities": ["gpu"],
                                }
                            ]
                        }
                    }
                },
                "healthcheck": {
                    "test": ["CMD", "python", "-c", HEALTH],
                    "interval": "15s",
                    "timeout": "5s",
                    "start_period": "120s",
                    "retries": 4,
                },
                "logging": {"options": {"max-size": "10m", "max-file": "3"}},
            }
        },
        "networks": {"modules": {"external": True, "name": network}},
    }
    return state, document


def validate_archive(path, expected):
    """Allow exactly the declared new image tag; verify included OCI blob hashes.

    Delta archives omit existing baseline blobs. No archive is extracted here.
    """
    index, seen = None, set()
    try:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                name = member.name
                if name in seen or not member.isfile():
                    raise ValueError("Duplicate or non-file OCI archive member")
                seen.add(name)
                stream = archive.extractfile(member)
                if name in ("index.json", "oci-layout"):
                    if member.size > 1024**2:
                        raise ValueError("OCI metadata is too large")
                    value = json.load(stream)
                    if name == "index.json":
                        index = value
                    continue
                if not re.fullmatch(r"blobs/sha256/[a-f0-9]{64}", name):
                    raise ValueError("Unexpected OCI archive member")
                checksum, small = hashlib.sha256(), bytearray()
                for chunk in iter(lambda stream=stream: stream.read(8 * 1024**2), b""):
                    checksum.update(chunk)
                    if member.size <= 2 * 1024**2:
                        small.extend(chunk)
                if checksum.hexdigest() != name.rsplit("/", 1)[1]:
                    raise ValueError("OCI blob digest mismatch")
                if small:
                    try:
                        value = json.loads(small)
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(value, dict):
                        for child in value.get("manifests", []):
                            if any(
                                key in child.get("annotations", {})
                                for key in (
                                    "io.containerd.image.name",
                                    "org.opencontainers.image.ref.name",
                                )
                            ):
                                raise ValueError(
                                    "Nested OCI image alias is not allowed"
                                )
    except (tarfile.TarError, KeyError, TypeError) as error:
        raise ValueError("Invalid OCI image archive") from error
    if index is None or "oci-layout" not in seen:
        raise ValueError("An OCI image archive is required")
    entries = index.get("manifests", [])
    if len(entries) != 1:
        raise ValueError("Archive must import only the independent neural image")
    entry = entries[0]
    annotations = entry.get("annotations", {})
    name = annotations.get("io.containerd.image.name", "").removeprefix(
        "docker.io/library/"
    )
    alias = annotations.get("org.opencontainers.image.ref.name", name)
    if (
        name != expected["name"]
        or entry.get("digest") != expected["id"]
        or alias not in (name, "docker.io/library/" + name, name.split(":", 1)[-1])
    ):
        raise ValueError("OCI archive identities differ from RELEASE.json")


@contextmanager
def state_lock(state):
    # Same lock file and protocol as the original GoodVision installer.
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
                            "GoodVision installation is busy; retry later"
                        ) from None
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Docker:
    def __init__(self, executable="docker"):
        self.executable = executable

    def run(self, args, **kwargs):
        return subprocess.run([self.executable, *args], check=True, **kwargs)

    def json(self, args):
        return json.loads(self.run(args, capture_output=True, text=True).stdout)

    def image(self, entry):
        found = self.json(["image", "inspect", entry["name"]])[0]
        if (
            found["Id"] not in {entry["id"], entry.get("config_id")}
            or found["Os"] != entry["os"]
            or found["Architecture"] != entry["architecture"]
        ):
            raise ValueError("Required image identity differs: " + entry["name"])
        return found

    def container(self):
        found = self.run(
            ["ps", "-aq", "--filter", "name=^/goodvision-neural$"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return self.json(["inspect", MODULE])[0] if found else None

    def project_members(self):
        found = self.run(
            ["ps", "-aq", "--filter", "label=com.docker.compose.project=" + MODULE],
            capture_output=True,
            text=True,
        ).stdout.split()
        return self.json(["inspect", *found]) if found else []


def check_owner(container, state):
    if container is not None:
        labels = container["Config"].get("Labels") or {}
        if (
            labels.get(LABEL) != str(state)
            or labels.get("com.docker.compose.project") != MODULE
            or labels.get("com.docker.compose.service") != "neural"
        ):
            raise ValueError(
                "goodvision-neural already belongs to another installation"
            )


def check_project(members, state):
    # Compose selects by project/service labels as well as container_name.
    # Refuse renamed or extra members before `up` can replace any of them.
    if len(members) > 1:
        raise ValueError("The neural Compose project contains extra containers")
    for container in members:
        check_owner(container, state)
        if container.get("Name") != "/" + MODULE:
            raise ValueError("The neural Compose project contains a renamed container")


def check_network_alias(docker, state, network, alias):
    info = docker.json(["network", "inspect", network])[0]
    identities = list((info.get("Containers") or {}).keys())
    if not identities:
        return
    for container in docker.json(["inspect", *identities]):
        aliases = (
            container.get("NetworkSettings", {})
            .get("Networks", {})
            .get(network, {})
            .get("Aliases")
            or []
        )
        if alias in aliases:
            try:
                check_owner(container, state)
                if container.get("Name") != "/" + MODULE:
                    raise ValueError("Unexpected neural container name")
            except ValueError as error:
                raise ValueError(
                    "Main's neural network alias is already used by another container: "
                    + alias
                ) from error


def compose_command(state):
    return [
        "compose",
        "--project-directory",
        str(state),
        "-p",
        MODULE,
        "-f",
        str(state / "compose.neural.json"),
    ]


def record(info, document, status, *, mode="oci-delta", actual_image=None):
    return {
        "module": MODULE,
        "version": info["version"],
        "image": info["image"],
        "status": status,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "compose": document,
        "gpu_inference": "not_verified",
        "install_mode": mode,
        "actual_image": actual_image,
    }


def verify_source(directory, info):
    files = info.get("source_files", [])
    required = {
        "source/workers/neural/Dockerfile",
        "source/workers/neural/healthcheck.py",
    }
    declared = {entry.get("file") for entry in files}
    if not required <= declared or len(declared) != len(files):
        raise ValueError("Offline build requires the declared neural source files")
    for entry in files:
        path = directory / entry["file"]
        if (
            not path.is_file()
            or not path.resolve().is_relative_to(directory.resolve())
            or path.stat().st_size != entry["bytes"]
            or digest(path) != entry["sha256"]
        ):
            raise ValueError("Offline build source differs: " + entry["file"])


def verify_built_image(found, baseline):
    layers = baseline.get("RootFS", {}).get("Layers", [])
    config = found.get("Config", {})
    labels = config.get("Labels") or {}
    if (
        not layers
        or found.get("RootFS", {}).get("Layers", [])[: len(layers)] != layers
        or found.get("Os") != "linux"
        or found.get("Architecture") != "amd64"
        or config.get("Entrypoint")
        != ["python", "-m", "goodvision_harness.neural_service"]
        or config.get("Cmd")
        != ["--host", "0.0.0.0", "--port", "8098", "--cache-dir", "/models/huggingface"]
        or labels.get("com.goodvision.module") != MODULE
        or labels.get("org.opencontainers.image.version") != "v2.0"
        or config.get("Healthcheck", {}).get("Test")
        != [
            "CMD",
            "python",
            "/opt/goodvision-neural/healthcheck.py",
        ]
    ):
        raise ValueError(
            "Offline build did not preserve the verified base and CLIP entrypoint"
        )
    return {
        "name": MODULE,
        "id": found["Id"],
        "os": found["Os"],
        "architecture": found["Architecture"],
    }


def install(
    goodvision_dir,
    directory,
    docker,
    *,
    timeout=900,
    host_port=28098,
    start_only=False,
    offline_build=False,
):
    info = release(directory)
    state, document = deployment(goodvision_dir, info, host_port)
    with state_lock(state):
        # Re-read under the same lock used by other module installers.
        state, document = deployment(goodvision_dir, info, host_port)
        compose = state / "compose.neural.json"
        marker = state / "neural-deployment.json"
        if compose.exists() or marker.exists():
            if (
                not compose.is_file()
                or not marker.is_file()
                or read(compose) != document
                or read(marker).get("compose") != document
            ):
                raise ValueError(
                    "Existing neural configuration differs; files left unchanged"
                )
        elif start_only:
            raise ValueError("Install the independent neural container first")
        engine = docker.json(["info", "--format", "{{json .}}"])
        if engine.get("OSType") != "linux" or engine.get("Architecture") not in (
            "amd64",
            "x86_64",
        ):
            raise ValueError("Use a Linux/amd64 Docker engine with NVIDIA GPU support")
        docker.run(["compose", "version"])
        check_owner(docker.container(), state)
        check_project(docker.project_members(), state)
        network = document["networks"]["modules"]["name"]
        alias = document["services"]["neural"]["networks"]["modules"]["aliases"][0]
        check_network_alias(docker, state, network, alias)
        mode, actual = "oci-delta", None
        if start_only:
            saved = read(marker)
            mode, actual = (
                saved.get("install_mode", "oci-delta"),
                saved.get("actual_image"),
            )
        elif offline_build:
            verify_source(directory, info)
            if len(info.get("requires", [])) != 1:
                raise ValueError("Offline build requires one verified V2.0 base image")
            base = info["requires"][0]
            baseline = docker.image(base)
            docker.run(
                [
                    "build",
                    "--network",
                    "none",
                    "--pull=false",
                    "--platform",
                    "linux/amd64",
                    "--build-arg",
                    "NEURAL_BASE_IMAGE=" + base["name"],
                    "-f",
                    str(directory / "source/workers/neural/Dockerfile"),
                    "-t",
                    info["image"]["name"],
                    str(directory / "source"),
                ]
            )
            docker.image(base)
            found = docker.json(["image", "inspect", info["image"]["name"]])[0]
            actual = verify_built_image(found, baseline)
            actual["name"] = info["image"]["name"]
            mode = "offline-build"
        else:
            archive = directory / info["image_archive"]["file"]
            metadata = info["image_archive"]
            if (
                not archive.is_file()
                or archive.stat().st_size != metadata["bytes"]
                or digest(archive) != metadata["sha256"]
            ):
                raise ValueError("Missing, incomplete or corrupt neural image archive")
            validate_archive(archive, info["image"])
            for entry in info.get("requires", []):
                docker.image(entry)
            # Existing tags of all other modules are excluded by validate_archive.
            docker.run(["load", "-i", str(archive)])
        if mode == "offline-build":
            if actual is None:
                raise ValueError("Installed offline build image identity is missing")
            docker.image(actual)
        else:
            docker.image(info["image"])
        expected = actual or info["image"]
        if not compose.exists():
            write_json(
                marker,
                record(info, document, "prepared", mode=mode, actual_image=actual),
            )
            write_json(compose, document)
        try:
            docker.run(
                [
                    *compose_command(state),
                    "up",
                    "-d",
                    "--no-deps",
                    "--pull",
                    "never",
                    "--wait",
                    "--wait-timeout",
                    str(timeout),
                    "neural",
                ]
            )
            container = docker.container()
            check_owner(container, state)
            if (
                container is None
                or container["Image"] not in {expected["id"], expected.get("config_id")}
                or container.get("State", {}).get("Health", {}).get("Status")
                != "healthy"
            ):
                raise ValueError(
                    "The neural container has not passed its model readiness "
                    "health check"
                )
            result = record(info, document, "healthy", mode=mode, actual_image=actual)
            result.update(
                container_id=container["Id"],
                gpu_inference="service_warmup_and_health_passed",
            )
            write_json(marker, result)
            print("Ready: goodvision-neural (CLIP " + info["version"] + ")")
        except (subprocess.SubprocessError, ValueError):
            write_json(
                marker,
                record(
                    info, document, "startup_failed", mode=mode, actual_image=actual
                ),
            )
            print(
                "CLIP startup failed; other modules were left running. "
                "Inspect with Status-Neural.cmd.",
                file=sys.stderr,
            )
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "start", "status", "plan"))
    parser.add_argument("--goodvision-dir", type=Path, required=True)
    parser.add_argument(
        "--release-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--host-port", type=int, default=28098)
    parser.add_argument("--wait-timeout", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--offline-build",
        action="store_true",
        help="Build offline from the verified installed V2.0 image; "
        "works with classic image stores",
    )
    args = parser.parse_args(argv)
    if args.wait_timeout < 1:
        raise ValueError("Wait timeout must be positive")
    directory = args.release_dir.resolve()
    info = release(directory)
    state, document = deployment(args.goodvision_dir, info, args.host_port)
    if args.action == "plan" or args.dry_run:
        print(
            json.dumps(
                {
                    "module": MODULE,
                    "version": info["version"],
                    "state": str(state),
                    "compose": document,
                    "requires": info.get("requires", []),
                    "image_archive": info["image_archive"],
                    "changes_existing_modules": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    docker = Docker(args.docker)
    if args.action == "status":
        container = docker.container()
        check_owner(container, state)
        result = {"module": MODULE, "installed": container is not None}
        if container is not None:
            marker = state / "neural-deployment.json"
            saved = read(marker) if marker.is_file() else {}
            expected = saved.get("actual_image") or info["image"]
            result.update(
                image_id=container["Image"],
                state=container["State"],
                expected_image=expected,
                install_mode=saved.get("install_mode"),
                image_matches=container["Image"]
                in {expected["id"], expected.get("config_id")},
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    install(
        args.goodvision_dir,
        directory,
        docker,
        timeout=args.wait_timeout,
        host_port=args.host_port,
        start_only=args.action == "start",
        offline_build=args.offline_build,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
