"""Repeatable Docker Compose updates for an installed Harness or Contour project.

Python 3.10+ standard library only. Run on the Docker host. Source builds use
the installed Compose build contexts; image-only deployments accept --image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

LABEL = "com.docker.compose."
BUSY = {"starting", "running", "stopping", "draining"}


class UpdateError(RuntimeError):
    """An actionable error without Docker output containing credentials."""


class ApiError(UpdateError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"Application check returned HTTP {status}.")


class Docker:
    def run(self, *args: str, cwd: Path | None = None, timeout=60, input=None) -> str:
        try:
            result = subprocess.run(
                ["docker", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
                input=input,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (OSError, subprocess.SubprocessError) as error:
            # Compose output and exec failures may contain environment/keys.
            raise UpdateError(
                f"Docker {args[0]} failed; check Docker locally."
            ) from error
        return result.stdout.strip()

    def json(self, *args: str, **kwargs):
        try:
            return json.loads(self.run(*args, **kwargs))
        except ValueError as error:
            raise UpdateError("Docker returned invalid JSON.") from error

    def api(self, identity: str, component: str, path: str, body=None, *, method=None):
        # Authentication stays inside the container, never in command arguments.
        code = (
            "import json,os,pathlib,urllib.request,urllib.error;"
            "headers={'Content-Type':'application/json','X-Fleet-Coordinator':'1'};"
        )
        if component == "contour":
            code += (
                "key=os.environ.get('CONTOUR_API_KEY') or pathlib.Path("
                "os.environ.get('CONTOUR_API_KEY_FILE','/run/secrets/node_key'))"
                ".read_text().strip();headers['Authorization']='Bearer '+key;"
            )
        port = {"contour": 8093, "engine": 8092}.get(component, 8080)
        port_expression = (
            "int(os.environ.get('HARNESS_PORT','8080'))"
            if component == "harness"
            else str(port)
        )
        code += (
            f"body={body!r};data=None if body is None else json.dumps(body).encode();"
            f"port={port_expression};"
            f"request=urllib.request.Request('http://127.0.0.1:'+str(port)+{path!r},"
            f"data=data,headers=headers,method={method!r});"
            "\ntry: print(urllib.request.urlopen(request,timeout=10).read().decode())"
            "\nexcept urllib.error.HTTPError as error: "
            "print(json.dumps({'goodvision_update_http_error':error.code}))"
        )
        result = self.json("exec", identity, "python", "-c", code, timeout=20)
        if "goodvision_update_http_error" in result:
            raise ApiError(result["goodvision_update_http_error"])
        return result

    def create_container(self, config, name, helper_image):
        # Docker Desktop exposes its Linux socket inside Linux containers. Send
        # settings through stdin, never command arguments or files on the host.
        code = (
            "import http.client,json,socket,sys,urllib.parse;"
            "request=json.load(sys.stdin);"
            "connection=http.client.HTTPConnection('localhost',timeout=150);"
            "connection.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);"
            "connection.sock.settimeout(150);"
            "connection.sock.connect('/var/run/docker.sock');"
            "connection.request('POST','/v1.45/containers/create?name='+"
            "urllib.parse.quote(request['name'],safe=''),"
            "json.dumps(request['config']),{'Content-Type':'application/json'});"
            "response=connection.getresponse();data=response.read();"
            "sys.exit(1) if response.status>=300 else print(data.decode())"
        )
        return self.json(
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--user",
            "0:0",
            "--label",
            "goodvision.update.helper=true",
            "--entrypoint",
            "python",
            "-v",
            "/var/run/docker.sock:/var/run/docker.sock",
            helper_image,
            "-c",
            code,
            input=json.dumps({"name": name, "config": config}),
            timeout=180,
        )["Id"]


@dataclass
class Deployment:
    component: str
    project: str
    directory: Path
    files: tuple[Path, ...]
    env_files: tuple[Path, ...]
    containers: dict[str, dict]
    config: dict = field(default_factory=dict)
    manual_names: dict[str, str] = field(default_factory=dict)

    @property
    def services(self):
        return sorted(self.containers)

    @property
    def state_dir(self):
        return self.directory / ".goodvision" / "docker-updates" / self.project

    @property
    def overlay(self):
        return self.state_dir / "active.compose.json"

    @property
    def record(self):
        return self.state_dir / "latest.json"

    def compose(self, *args, active=True):
        command = [
            "compose",
            "--project-name",
            self.project,
            "--project-directory",
            str(self.directory),
        ]
        for path in self.env_files:
            command += ["--env-file", str(path)]
        for path in self.files:
            command += ["-f", str(path)]
        if active and self.overlay.is_file():
            command += ["-f", str(self.overlay)]
        return [*command, *args]

    def fingerprint(self):
        # Store a digest, never copies of .env or resolved config with secrets.
        digest = hashlib.sha256()
        for path in (*self.files, *self.env_files):
            digest.update(str(path).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()


def selected_service(component, service):
    return (
        bool(re.fullmatch(r"harness(?:-\d+)?", service))
        if component == "harness"
        else service in {"contour", "engine"}
    )


def check_host(docker):
    context = docker.json("context", "inspect")[0]
    endpoint = context["Endpoints"]["docker"]["Host"]
    if os.environ.get("DOCKER_HOST") and not os.environ.get("DOCKER_CONTEXT"):
        endpoint = os.environ["DOCKER_HOST"]
    if not endpoint.startswith(("npipe://", "unix://")):
        raise UpdateError("Run the updater on the Docker host using a local context.")
    if docker.run("info", "--format", "{{.OSType}}") != "linux":
        raise UpdateError("Linux Docker containers are required.")


def discover(
    docker,
    component,
    project=None,
    env_files=(),
    *,
    runtime=False,
    state_directory=None,
):
    check_host(docker)
    if not runtime:
        docker.run("compose", "version")
    ids = docker.run("ps", "-aq", "--filter", "label=" + LABEL + "project").split()
    installed = docker.json("inspect", *ids) if ids else []
    candidates = []
    for container in installed:
        labels = container["Config"].get("Labels") or {}
        if (
            selected_service(component, labels.get(LABEL + "service", ""))
            and labels.get(LABEL + "oneoff", "False").lower() != "true"
            and labels.get("goodvision.update.helper") != "true"
            and "-before-" not in container["Name"]
            and (project is None or labels.get(LABEL + "project") == project)
        ):
            candidates.append(container)
    projects = {c["Config"]["Labels"][LABEL + "project"] for c in candidates}
    if len(projects) != 1:
        raise UpdateError(
            "Select one installed project with --project; found: "
            + ", ".join(sorted(projects))
        )
    project = projects.pop()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project):
        raise UpdateError("Unsupported Compose project name.")
    if runtime:
        containers = {c["Config"]["Labels"][LABEL + "service"]: c for c in candidates}
        if len(containers) != len(candidates):
            raise UpdateError(
                "Scaled replicas are unsupported; use separate harness-N lanes."
            )
        if component == "contour" and set(containers) != {"contour", "engine"}:
            raise UpdateError(
                "A GPU Contour installation needs both contour and engine."
            )
        locations = {
            c["Config"]["Labels"].get(LABEL + "project.working_dir") for c in candidates
        }
        if len(locations) != 1:
            raise UpdateError(
                "Selected containers have different installation folders."
            )
        location = locations.pop()
        if location and Path(location).is_dir():
            directory = Path(location).resolve()
        else:
            if state_directory is None:
                raise UpdateError(
                    "An update state folder is required for this installation."
                )
            directory = Path(state_directory).resolve()
        configurations = {
            c["Config"]["Labels"].get(LABEL + "project.config_files", "")
            for c in candidates
        }
        if len(configurations) != 1:
            raise UpdateError(
                "Selected containers have different Compose configurations."
            )
        files = tuple(Path(p).resolve() for p in configurations.pop().split(",") if p)
        files = tuple(
            p
            for p in files
            if p
            != directory
            / ".goodvision"
            / "docker-updates"
            / project
            / "active.compose.json"
        )
        if not all(p.is_file() for p in files):
            files = ()
        recorded = {
            c["Config"]["Labels"].get(LABEL + "project.environment_file", "")
            for c in candidates
        }
        if len(recorded) != 1:
            raise UpdateError("Selected containers have different environment files.")
        env_files = tuple(Path(p).resolve() for p in recorded.pop().split(",") if p)
        if not env_files and (directory / ".env").is_file():
            env_files = (directory / ".env",)
        if not all(p.is_file() for p in env_files):
            # The live runtime can still be updated, but an incomplete launcher
            # cannot be reconstructed safely to determine its image reference.
            files, env_files = (), ()
        # Customer image updates clone the inspected Docker configuration. They
        # also support older docker-run installs carrying only workload labels.
        return Deployment(component, project, directory, files, env_files, containers)
    locations = {
        (
            c["Config"]["Labels"].get(LABEL + "project.working_dir"),
            c["Config"]["Labels"].get(LABEL + "project.config_files"),
        )
        for c in candidates
    }
    if len(locations) != 1:
        raise UpdateError("Selected containers have different Compose configurations.")
    directory, config_files = locations.pop()
    if not directory or not config_files:
        raise UpdateError("Installed Compose paths are missing.")
    directory = Path(directory).resolve()
    overlay = (
        directory / ".goodvision" / "docker-updates" / project / "active.compose.json"
    )
    files = tuple(
        Path(p).resolve()
        for p in config_files.split(",")
        if Path(p).resolve() != overlay
    )
    if not files or not directory.is_dir() or any(not p.is_file() for p in files):
        raise UpdateError("Installed Compose files are unavailable on this host.")
    if not env_files:
        recorded = {
            c["Config"]["Labels"].get(LABEL + "project.environment_file", "")
            for c in candidates
        }
        if len(recorded) != 1:
            raise UpdateError("Selected containers have different environment files.")
        value = recorded.pop()
        env_files = (
            value.split(",")
            if value
            else [directory / ".env"]
            if (directory / ".env").is_file()
            else []
        )
    env_files = tuple(Path(p).resolve() for p in env_files)
    if any(not p.is_file() for p in env_files):
        raise UpdateError("Installed environment file is unavailable; use --env-file.")
    containers = {c["Config"]["Labels"][LABEL + "service"]: c for c in candidates}
    if len(containers) != len(candidates):
        raise UpdateError(
            "Scaled replicas are unsupported; use separate harness-N lanes."
        )
    if component == "contour" and set(containers) != {"contour", "engine"}:
        raise UpdateError("A GPU Contour installation needs both contour and engine.")
    deployment = Deployment(component, project, directory, files, env_files, containers)
    deployment.config = docker.json(
        *deployment.compose("config", "--format", "json"), cwd=directory
    )
    if not set(containers) <= deployment.config["services"].keys():
        raise UpdateError("Installed services are missing from Compose configuration.")
    return deployment


def check_config(docker, deployment):
    for service, container in deployment.containers.items():
        output = docker.run(
            *deployment.compose("config", "--hash", service), cwd=deployment.directory
        )
        expected = container["Config"]["Labels"].get(LABEL + "config-hash")
        if not expected or output.split()[-1:] != [expected]:
            raise UpdateError(
                "Compose/environment differs from the installed service "
                + service
                + "; restore deployment settings before updating."
            )


def configuration_fingerprint(docker, deployment):
    config = docker.json(
        *deployment.compose("config", "--format", "json"), cwd=deployment.directory
    )
    for service in deployment.services:
        config["services"][service].pop("image", None)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def assert_idle(docker, deployment, *, recovery=False):
    accepting = None
    for service, container in deployment.containers.items():
        if not container["State"].get("Running"):
            if recovery:
                continue
            raise UpdateError("Start the installed services before updating.")
        if container["State"].get("Paused"):
            raise UpdateError("Unpause the installed services before updating.")
        if deployment.component == "harness":
            status = docker.api(container["Id"], "harness", "/v1/status")
            settings = docker.api(container["Id"], "harness", "/v1/deployment/settings")
            if (
                status.get("analysis_state") in BUSY
                or status.get("pipeline_state") in BUSY
                or status.get("processing_cleanup") == "closing"
            ):
                raise UpdateError(
                    "Stop all analysis and wait for cleanup before updating."
                )
            if settings.get("autostart") is not False:
                raise UpdateError("Disable automatic analysis startup before updating.")
        elif service == "contour":
            status = docker.api(container["Id"], "contour", "/api/status")
            engine_stopped = recovery and not deployment.containers["engine"][
                "State"
            ].get("Running")
            if (
                not engine_stopped
                and (
                    type(status.get("active_sessions")) is not int
                    or status["active_sessions"] != 0
                )
            ) or status.get("requests", {}).get("in_flight") != 0:
                raise UpdateError(
                    "Stop all Contour sessions and wait for requests to finish."
                )
            accepting = status.get("accepting_sessions")
            if type(accepting) is not bool:
                raise UpdateError("Cannot determine Contour admission state.")
    return accepting


def current_containers(docker, deployment):
    if deployment.manual_names:
        values = docker.json("inspect", *deployment.manual_names.values())
        by_name = {c["Name"].lstrip("/"): c for c in values}
        if set(by_name) != set(deployment.manual_names.values()):
            raise UpdateError("Selected containers disappeared during update.")
        return {s: by_name[name] for s, name in deployment.manual_names.items()}
    ids = docker.run(
        "ps", "-aq", "--filter", "label=" + LABEL + "project=" + deployment.project
    ).split()
    values = docker.json("inspect", *ids) if ids else []
    result = {}
    for value in values:
        labels = value["Config"].get("Labels") or {}
        service = labels.get(LABEL + "service")
        if (
            service in deployment.services
            and labels.get(LABEL + "oneoff", "False").lower() != "true"
            and "-before-" not in value["Name"]
        ):
            if service in result:
                raise UpdateError("Unexpected replicas appeared during update.")
            result[service] = value
    if set(result) != set(deployment.services):
        raise UpdateError("Selected services disappeared during update.")
    return result


def wait_ready(docker, deployment, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        containers = current_containers(docker, deployment)
        if any(
            not c["State"].get("Running")
            or c["State"].get("OOMKilled")
            or c["State"].get("Restarting")
            for c in containers.values()
        ):
            raise UpdateError(
                "A replacement service stopped, restarted or ran out of memory."
            )
        try:
            for service, container in containers.items():
                if deployment.component == "harness":
                    docker.api(container["Id"], "harness", "/v1/ui/info")
                    status = docker.api(container["Id"], "harness", "/v1/status")
                    settings = docker.api(
                        container["Id"], "harness", "/v1/deployment/settings"
                    )
                    if (
                        status.get("analysis_state") in BUSY
                        or status.get("pipeline_state") in BUSY
                        or status.get("processing_cleanup") == "closing"
                        or settings.get("autostart") is not False
                    ):
                        raise UpdateError("Replacement harness did not remain idle.")
                elif service == "contour":
                    # Close admission as soon as the new gateway responds,
                    # before waiting for the engine to finish GPU warmup.
                    docker.api(
                        container["Id"],
                        "contour",
                        "/api/admission",
                        {"accepting_sessions": False},
                    )
                    if (
                        docker.api(container["Id"], "contour", "/api/status").get(
                            "gpu_ready"
                        )
                        is not True
                    ):
                        raise UpdateError("GPU warmup is pending.")
                    status = docker.api(container["Id"], "contour", "/api/status")
                    if (
                        status.get("active_sessions") != 0
                        or status.get("requests", {}).get("in_flight") != 0
                    ):
                        raise UpdateError("Contour sessions appeared during startup.")
            return
        except UpdateError:
            time.sleep(2)
    raise UpdateError("Replacement services did not become ready before the timeout.")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def update_lock(deployment):
    deployment.state_dir.mkdir(parents=True, exist_ok=True)
    path = deployment.state_dir / "update.lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise UpdateError(
            "An update lock exists. If the previous updater exited, remove "
            + str(path)
            + " before recovery."
        ) from error
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def replace_services(docker, deployment, images, timeout):
    services = {}
    if deployment.overlay.is_file():
        services = json.loads(deployment.overlay.read_text(encoding="utf-8"))[
            "services"
        ]
    services.update({s: {"image": image} for s, image in images.items()})
    write_json(
        deployment.overlay,
        {"services": services},
    )
    docker.run(
        *deployment.compose("stop", *deployment.services),
        cwd=deployment.directory,
        timeout=180,
    )
    docker.run(
        *deployment.compose(
            "up",
            "-d",
            "--no-build",
            "--pull",
            "never",
            "--force-recreate",
            "--no-deps",
            *deployment.services,
        ),
        cwd=deployment.directory,
        timeout=180,
    )
    wait_ready(docker, deployment, timeout)


def image_id(docker, reference):
    return docker.json("image", "inspect", reference)[0]["Id"]


def launcher_references(docker, deployment, candidates, *, deduplicate=True):
    config = (
        docker.json(
            *deployment.compose("config", "--format", "json", active=False),
            cwd=deployment.directory,
        )
        if deployment.files
        else {
            "services": {
                service: {
                    "image": (container["Config"].get("Labels") or {}).get(
                        "goodvision.update.launcher_image",
                        container["Config"].get("Image"),
                    )
                }
                for service, container in deployment.containers.items()
            }
        }
    )
    references = {}
    service_references = {}
    for service in deployment.services:
        reference = config["services"][service].get("image")
        if not reference or reference.startswith("sha256:") or "@" in reference:
            raise UpdateError("The installed launcher needs a supported image tag.")
        if (
            reference in references
            and candidates[service] != candidates[references[reference]]
        ):
            raise UpdateError("Selected services require conflicting launcher images.")
        references[reference] = service
        service_references[service] = reference
    ids = docker.run("ps", "-aq").split()
    selected_ids = {c["Id"] for c in deployment.containers.values()}
    backup_ids = set()
    runtime_record = deployment.state_dir / "runtime-latest.json"
    if runtime_record.is_file():
        record = json.loads(runtime_record.read_text("utf-8"))
        if record.get("project") == deployment.project:
            backup_ids = {
                item["old_id"] for item in record.get("containers", {}).values()
            }
            backup_ids.update(record.get("retained_backups", []))
    for container in docker.json("inspect", *ids) if ids else []:
        labels = container["Config"].get("Labels") or {}
        own_backup = (
            container["Id"] in backup_ids
            and (
                (
                    labels.get(LABEL + "project") == deployment.project
                    and labels.get(LABEL + "service") in deployment.services
                )
                or (
                    deployment.manual_names
                    and any(
                        container["Name"].lstrip("/").startswith(name + "-before-")
                        for name in deployment.manual_names.values()
                    )
                )
            )
            and "-before-" in container["Name"]
            and not container["State"].get("Running")
        )
        if (
            container["Id"] not in selected_ids
            and not own_backup
            and (
                container["Config"].get("Image") in references
                or labels.get("goodvision.update.launcher_image") in references
            )
        ):
            raise UpdateError("Another installation shares these launcher image tags.")
    return (
        {service: reference for reference, service in references.items()}
        if deduplicate
        else service_references
    )


def tag_launcher_images(docker, references, images):
    for service, reference in references.items():
        docker.run("image", "tag", images[service], reference)


def update(
    docker,
    deployment,
    *,
    images=None,
    check_only=False,
    rollback=False,
    timeout=900,
    persist_references=False,
    lock_held=False,
):
    images = images or {}
    if images and rollback:
        raise UpdateError("--image cannot be combined with --rollback.")
    if images and set(images) != set(deployment.services):
        raise UpdateError(
            "--image must specify every selected service: "
            + ", ".join(deployment.services)
        )
    previous = None
    fingerprint = deployment.fingerprint()
    configuration = configuration_fingerprint(docker, deployment)
    original_ids = {s: c["Id"] for s, c in deployment.containers.items()}
    if rollback:
        if not deployment.record.is_file():
            raise UpdateError("No saved update exists for rollback.")
        previous = json.loads(deployment.record.read_text(encoding="utf-8"))
        if (
            previous["fingerprint"] != fingerprint
            or previous.get("configuration") != configuration
            or previous["component"] != deployment.component
            or set(previous["before"]) != set(deployment.services)
        ):
            raise UpdateError(
                "Compose or environment files changed since the saved update."
            )
        if any(
            c["Image"] not in (previous["before"][s], previous["after"][s])
            for s, c in deployment.containers.items()
        ):
            raise UpdateError("A newer image was installed outside this updater.")
        candidates = {s: image_id(docker, i) for s, i in previous["before"].items()}
    else:
        check_config(docker, deployment)
        if images:
            candidates = {s: image_id(docker, i) for s, i in images.items()}
        else:
            config = docker.json(
                *deployment.compose("config", "--format", "json", active=False),
                cwd=deployment.directory,
            )
            if any(not config["services"][s].get("build") for s in deployment.services):
                raise UpdateError(
                    "This deployment has no build definition; load candidate images "
                    "and pass --image for every service."
                )
            candidates = {}
    references = previous.get("references", {}) if rollback else {}
    if persist_references:
        references = launcher_references(docker, deployment, candidates)
    accepting = assert_idle(docker, deployment, recovery=rollback)
    if rollback and accepting is None:
        accepting = previous.get("accepting_sessions")
    print(
        "Selected " + deployment.project + ": " + ", ".join(deployment.services),
        flush=True,
    )
    if check_only:
        print("Checks passed. No images or services were changed.", flush=True)
        return
    with nullcontext() if lock_held else update_lock(deployment):
        if not rollback and not images:
            print(
                "Building services while the installed version stays running...",
                flush=True,
            )
            docker.run(
                *deployment.compose("build", *deployment.services, active=False),
                cwd=deployment.directory,
                timeout=7200,
            )
            candidates = {
                s: image_id(
                    docker,
                    config["services"][s].get("image", deployment.project + "-" + s),
                )
                for s in deployment.services
            }
        # Reinspect after the potentially long build. Never stop a new analysis.
        deployment.containers = current_containers(docker, deployment)
        if original_ids != {s: c["Id"] for s, c in deployment.containers.items()}:
            raise UpdateError(
                "Installed containers changed while preparing the update."
            )
        if (
            fingerprint != deployment.fingerprint()
            or configuration != configuration_fingerprint(docker, deployment)
        ):
            raise UpdateError(
                "Deployment configuration changed while preparing images."
            )
        if not rollback:
            check_config(docker, deployment)
        accepting_now = assert_idle(docker, deployment, recovery=rollback)
        if accepting_now is not None:
            accepting = accepting_now
        before = {s: c["Image"] for s, c in deployment.containers.items()}
        if not rollback and candidates == before:
            if persist_references:
                tag_launcher_images(docker, references, candidates)
                if deployment.record.is_file():
                    saved = json.loads(deployment.record.read_text(encoding="utf-8"))
                    if saved.get("after") == candidates:
                        saved["references"] = references
                        write_json(deployment.record, saved)
            print("Selected services already use these images.", flush=True)
            return
        record = {
            "component": deployment.component,
            "project": deployment.project,
            "fingerprint": fingerprint,
            "configuration": configuration,
            "before": before,
            "after": candidates,
            "accepting_sessions": accepting,
            "status": "prepared",
            "references": references,
        }
        # Keep every previous image reachable even when a build replaces its tag.
        transaction = uuid.uuid4().hex[:12]
        for service, identity in before.items():
            docker.run(
                "image",
                "tag",
                identity,
                f"goodvision-update-backup/{deployment.project}-{service}:{transaction}",
            )
        paused = False
        try:
            if deployment.component == "contour":
                gateway = deployment.containers["contour"]
                if gateway["State"].get("Running"):
                    paused = True
                    docker.api(
                        gateway["Id"],
                        "contour",
                        "/api/admission",
                        {"accepting_sessions": False},
                    )
                    assert_idle(docker, deployment, recovery=rollback)
            if not rollback:
                write_json(deployment.record, record)
            try:
                print(
                    "Replacing selected services and checking readiness...", flush=True
                )
                replace_services(docker, deployment, candidates, timeout)
                tag_launcher_images(docker, references, candidates)
            except (Exception, KeyboardInterrupt) as error:
                print("Startup failed. Restoring the previous images...", flush=True)
                try:
                    replace_services(docker, deployment, before, timeout)
                    tag_launcher_images(docker, references, before)
                except (Exception, KeyboardInterrupt) as recovery_error:
                    raise UpdateError(
                        "Recovery did not become ready. Keep the saved state and "
                        "run --rollback after checking Docker."
                    ) from recovery_error
                if not rollback:
                    record["status"] = "restored"
                    write_json(deployment.record, record)
                raise UpdateError(
                    "Update failed; previous images were restored."
                ) from error
            if not rollback:
                record["status"] = "ready"
                write_json(deployment.record, record)
        finally:
            if deployment.component == "contour" and (paused or rollback):
                failure_pending = sys.exc_info()[0] is not None
                try:
                    gateway = current_containers(docker, deployment)["contour"]
                    docker.api(
                        gateway["Id"],
                        "contour",
                        "/api/admission",
                        {"accepting_sessions": accepting},
                    )
                except UpdateError as admission_error:
                    message = (
                        "Could not restore Contour admission; check the node console."
                    )
                    if not failure_pending:
                        raise UpdateError(message) from admission_error
                    print(message, file=sys.stderr)
        print(
            "Previous version restored."
            if rollback
            else "Update ready. Analysis remains stopped.",
            flush=True,
        )
        print(
            "Keep this overlay in subsequent Compose commands: "
            + str(deployment.overlay),
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("harness", "contour"))
    parser.add_argument("--project")
    parser.add_argument("--env-file", type=Path, action="append", default=[])
    parser.add_argument("--image", metavar="SERVICE=IMAGE", action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        images = {}
        for assignment in args.image:
            service, separator, image = assignment.partition("=")
            if not separator or not service or not image or service in images:
                raise UpdateError("Use unique --image SERVICE=IMAGE assignments.")
            images[service] = image
        if args.timeout < 1:
            raise UpdateError("--timeout must be positive.")
        docker = Docker()
        deployment = discover(docker, args.component, args.project, args.env_file)
        update(
            docker,
            deployment,
            images=images,
            check_only=args.check_only,
            rollback=args.rollback,
            timeout=args.timeout,
        )
        return 0
    except (UpdateError, OSError, ValueError, KeyError) as error:
        print("Update stopped: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
