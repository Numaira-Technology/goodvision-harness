"""Resolve an explicitly supplied Docker name without guessing a workload."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

if __package__:
    from .update_docker import (
        LABEL,
        Deployment,
        UpdateError,
        check_host,
        discover,
        selected_service,
    )
else:
    from update_docker import (
        LABEL,
        Deployment,
        UpdateError,
        check_host,
        discover,
        selected_service,
    )


class NameLookupError(UpdateError):
    """A customer-readable lookup problem."""


def running(container):
    return container["State"].get("Running") and not container["State"].get("Paused")


def verify_harness(docker, container):
    if not running(container):
        return False
    try:
        info = docker.api(container["Id"], "harness", "/v1/ui/info")
        status = docker.api(container["Id"], "harness", "/v1/status")
        settings = docker.api(container["Id"], "harness", "/v1/deployment/settings")
        window = info.get("window", {})
        return (
            info.get("configured") is True
            and {"event", "face", "multi"} <= set(info.get("analysis_modes", []))
            and info.get("face", {}).get("engine") == "local"
            and all(
                type(window.get(k)) is int and window[k] > 0
                for k in ("size_frames", "stride_frames")
            )
            and isinstance(status.get("analysis_state"), str)
            and type(settings.get("autostart")) is bool
        )
    except (UpdateError, AttributeError, TypeError):
        return False


def verify_contour(docker, container):
    if not running(container):
        return False
    try:
        status = docker.api(container["Id"], "contour", "/api/status")
        return (
            status.get("service") == "GoodVision Contour"
            and status.get("model_id") == "goodvision-contour-1"
            and status.get("mode") == "connected"
            and type(status.get("accepting_sessions")) is bool
            and isinstance(status.get("requests"), dict)
        )
    except (UpdateError, AttributeError):
        return False


def verify_engine(docker, container):
    if not running(container):
        return False
    try:
        status = docker.api(container["Id"], "engine", "/health/perception/sam_live")
        return (
            status.get("backend") == "sam3.1_video"
            and status.get("enabled") is True
            and type(status.get("gpu_ready")) is bool
        )
    except (UpdateError, AttributeError):
        return False


def paired_engine(docker, gateway, installed):
    # Resolve the gateway's configured peer inside its own network namespace.
    # Environment/key values never leave that container or enter command args.
    code = (
        "import json,os,socket,urllib.parse;"
        "url=urllib.parse.urlparse(os.environ.get('CONTOUR_ENGINE_URL',''));"
        "assert url.scheme=='http' and url.hostname and url.port==8092;"
        "print(json.dumps(sorted({r[4][0] for r in "
        "socket.getaddrinfo(url.hostname,url.port,type=socket.SOCK_STREAM)})))"
    )
    try:
        addresses = set(docker.json("exec", gateway["Id"], "python", "-c", code))
    except (UpdateError, TypeError):
        return None
    networks = set(gateway.get("NetworkSettings", {}).get("Networks", {}))
    matches = []
    for container in installed:
        if container["Id"] == gateway["Id"]:
            continue
        endpoints = container.get("NetworkSettings", {}).get("Networks", {})
        if any(
            name in networks
            and {endpoint.get("IPAddress"), endpoint.get("GlobalIPv6Address")}
            & addresses
            for name, endpoint in endpoints.items()
        ) and verify_engine(docker, container):
            matches.append(container)
    return matches[0] if len(matches) == 1 else None


def inventory(docker):
    ids = docker.run("ps", "-aq").split()
    return [
        c
        for c in (docker.json("inspect", *ids) if ids else [])
        if (c["Config"].get("Labels") or {}).get("goodvision.update.helper") != "true"
        and (c["Config"].get("Labels") or {}).get(LABEL + "oneoff", "False").lower()
        != "true"
        and "-before-" not in c["Name"]
    ]


def proposal(docker, component, container, installed, state_directory):
    labels = container["Config"].get("Labels") or {}
    service = labels.get(LABEL + "service", "")
    project = labels.get(LABEL + "project")
    if project and selected_service(component, service):
        deployment = discover(
            docker, component, project, runtime=True, state_directory=state_directory
        )
        if component == "harness":
            valid = all(
                verify_harness(docker, c) for c in deployment.containers.values()
            )
        else:
            valid = verify_contour(
                docker, deployment.containers["contour"]
            ) and verify_engine(docker, deployment.containers["engine"])
        if not valid:
            return None
        selection = {
            "component": component,
            "project": project,
            "ids": {s: c["Id"] for s, c in deployment.containers.items()},
        }
    else:
        if component == "harness":
            if not verify_harness(docker, container):
                return None
            containers = {"harness": container}
        else:
            if not verify_contour(docker, container):
                return None
            engine = paired_engine(docker, container, installed)
            if engine is None:
                return None
            containers = {"contour": container, "engine": engine}
        selection = {
            "component": component,
            "names": {s: c["Name"].lstrip("/") for s, c in containers.items()},
            "ids": {s: c["Id"] for s, c in containers.items()},
        }
    return {"selection": selection, "label": container["Name"].lstrip("/")}


def lookup_named(docker, component, name, *, state_directory):
    name = name.strip().lstrip("/")
    if component not in {"harness", "contour"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,1023}", name
    ):
        raise NameLookupError(
            "Enter the container or image name shown in Docker Desktop."
        )
    check_host(docker)
    installed = inventory(docker)
    exact = [c for c in installed if c["Name"].lstrip("/") == name]
    candidates = exact
    if not exact:
        try:
            identity = docker.json("image", "inspect", name)[0]["Id"]
        except (UpdateError, IndexError, KeyError):
            identity = None
        candidates = [
            c
            for c in installed
            if c["Config"].get("Image") == name or c["Image"] == identity
        ]
    choices, seen = [], set()
    for container in candidates:
        try:
            choice = proposal(docker, component, container, installed, state_directory)
        except UpdateError:
            choice = None
        matches = [choice] if choice else []
        if not matches and component == "contour" and verify_engine(docker, container):
            # An explicitly supplied engine name can locate its gateway too,
            # using the same proven network relationship rather than a name guess.
            for gateway in installed:
                if not verify_contour(docker, gateway):
                    continue
                engine = paired_engine(docker, gateway, installed)
                if engine and engine["Id"] == container["Id"]:
                    try:
                        choice = proposal(
                            docker, component, gateway, installed, state_directory
                        )
                    except UpdateError:
                        choice = None
                    if choice:
                        matches.append(choice)
        for choice in matches:
            key = json.dumps(choice["selection"], sort_keys=True)
            if key not in seen:
                choices.append(choice)
                seen.add(key)
    return sorted(choices, key=lambda c: c["label"])


def named_deployment(docker, component, selection, *, state_directory):
    if not isinstance(selection, dict) or selection.get("component") != component:
        raise NameLookupError(
            "Choose a matching GoodVision installation and try again."
        )
    check_host(docker)
    if selection.get("project"):
        deployment = discover(
            docker,
            component,
            selection["project"],
            runtime=True,
            state_directory=state_directory,
        )
        valid = (
            all(verify_harness(docker, c) for c in deployment.containers.values())
            if component == "harness"
            else verify_contour(docker, deployment.containers["contour"])
            and verify_engine(docker, deployment.containers["engine"])
        )
    else:
        names = selection.get("names", {})
        roles = {"harness"} if component == "harness" else {"contour", "engine"}
        if set(names) != roles or any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", n) for n in names.values()
        ):
            raise NameLookupError(
                "Choose a matching GoodVision installation and try again."
            )
        values = docker.json("inspect", *names.values())
        by_name = {c["Name"].lstrip("/"): c for c in values}
        if set(by_name) != set(names.values()):
            raise NameLookupError(
                "That container changed. Find your installation again."
            )
        containers = {s: by_name[n] for s, n in names.items()}
        if component == "harness":
            valid = verify_harness(docker, containers["harness"])
        else:
            engine = paired_engine(docker, containers["contour"], inventory(docker))
            valid = (
                verify_contour(docker, containers["contour"])
                and verify_engine(docker, containers["engine"])
                and engine is not None
                and engine["Id"] == containers["engine"]["Id"]
            )
        key = json.dumps({"component": component, "names": names}, sort_keys=True)
        project = "manual-" + hashlib.sha256(key.encode()).hexdigest()[:16]
        location = next(iter(containers.values()))["Config"].get("Labels") or {}
        directory = location.get(LABEL + "project.working_dir")
        directory = (
            Path(directory)
            if directory and Path(directory).is_dir()
            else Path(state_directory)
        )
        deployment = Deployment(
            component, project, directory, (), (), containers, manual_names=dict(names)
        )
    if not valid:
        raise NameLookupError(
            "Open GoodVision and Docker Desktop, then find the installation again."
        )
    if selection.get("ids") != {s: c["Id"] for s, c in deployment.containers.items()}:
        raise NameLookupError("That container changed. Find your installation again.")
    return deployment
