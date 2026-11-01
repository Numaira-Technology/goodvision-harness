"""Offline Docker Engine updater. Runs inside the installed Linux image.

Only the selected supported installation is touched. Application data and model
volumes are retained. Successful updates remove the replaced containers.
"""

from __future__ import annotations

import argparse
import copy
import http.client
import json
import re
import socket
import struct
import time
from pathlib import Path
from urllib.parse import quote

if __package__:
    from .release_metadata import read_metadata, update_metadata
    from .verify_installation import verify_installation
else:
    from release_metadata import read_metadata, update_metadata
    from verify_installation import verify_installation


class EngineConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect("/var/run/docker.sock")


class Engine:
    def request(self, method, path, body=None, raw=False):
        connection = EngineConnection("localhost", timeout=150)
        try:
            data = None if body is None else json.dumps(body).encode()
            connection.request(
                method, "/v1.45" + path, data, {"Content-Type": "application/json"}
            )
            response = connection.getresponse()
            content = response.read()
            if response.status >= 300:
                # Engine errors may contain secret environment or mount content.
                raise RuntimeError(
                    f"Docker {method} {path.split('?')[0]}: HTTP {response.status}"
                )
            return content if raw else json.loads(content) if content else None
        finally:
            connection.close()

    def inspect(self, identity):
        return self.request("GET", f"/containers/{quote(identity, safe='')}/json")

    def api(self, identity, path, method="GET", body=None):
        code = (
            "import json,urllib.request;"
            f"data={body!r};"
            "data=None if data is None else json.dumps(data).encode();"
            f"req=urllib.request.Request('http://127.0.0.1:8080{path}',data=data,method={method!r},"
            "headers={'Content-Type':'application/json','X-Fleet-Coordinator':'1'});"
            "print(urllib.request.urlopen(req,timeout=15).read().decode())"
        )
        execution = self.request(
            "POST",
            f"/containers/{identity}/exec",
            {
                "AttachStdout": True,
                "AttachStderr": True,
                "Cmd": ["python", "-c", code],
            },
        )["Id"]
        wire = self.request(
            "POST",
            f"/exec/{execution}/start",
            {"Detach": False, "Tty": False},
            raw=True,
        )
        if self.request("GET", f"/exec/{execution}/json")["ExitCode"] != 0:
            raise RuntimeError(f"Application check failed: {path}")
        output = bytearray()
        while wire:
            stream, length = wire[0], struct.unpack(">I", wire[4:8])[0]
            if stream == 1:
                output.extend(wire[8 : 8 + length])
            wire = wire[8 + length :]
        return json.loads(output)


def replacement_config(original, image, *, continuous_playback_vlm=False):
    config = copy.deepcopy(original["Config"])
    config["Image"] = image
    if continuous_playback_vlm:
        environment = dict(value.split("=", 1) for value in config.get("Env", []))
        environment.update(
            LOCAL_CAMERA_TARGET_FPS="15",
            SCENE_ENGINE_MODE="active",
            SCENE_INDEPENDENT_VLM="true",
        )
        config["Env"] = [f"{key}={value}" for key, value in environment.items()]
    # These are image metadata, not ContainerCreate fields.
    for key in ("OnBuild", "Shell"):
        config.pop(key, None)
    host = copy.deepcopy(original["HostConfig"])
    # Docker normalizes dynamic ports; preserve the actual host bindings.
    for port, bindings in (original["NetworkSettings"].get("Ports") or {}).items():
        if bindings:
            host.setdefault("PortBindings", {})[port] = bindings
    # Pin anonymous volume names too. Never let Docker allocate empty replacements.
    bound = set()
    for item in host.get("Binds") or []:
        tail = item.rsplit(":", 1)
        bound.add(tail[-1] if tail[-1].startswith("/") else tail[0].rsplit(":", 1)[-1])
    explicit = {item["Target"] for item in host.get("Mounts") or []}
    for mount in original.get("Mounts", []):
        if mount["Type"] == "volume" and mount["Destination"] not in bound | explicit:
            if host.get("Mounts") is None:
                host["Mounts"] = []
            host["Mounts"].append(
                {
                    "Type": "volume",
                    "Source": mount["Name"],
                    "Target": mount["Destination"],
                    "ReadOnly": not mount["RW"],
                }
            )
    networks = {}
    for name, endpoint in original["NetworkSettings"]["Networks"].items():
        if endpoint.get("IPAMConfig") or endpoint.get("Links"):
            raise RuntimeError(
                "Static container IPs or legacy links require a custom update."
            )
        networks[name] = {
            key: copy.deepcopy(endpoint[key])
            for key in ("IPAMConfig", "Links", "Aliases", "DriverOpts", "MacAddress")
            if endpoint.get(key)
        }
        # Automatically assigned container IDs must not become lasting aliases.
        if "Aliases" in networks[name]:
            networks[name]["Aliases"] = [
                alias
                for alias in networks[name]["Aliases"]
                if alias not in (original["Id"], original["Id"][:12])
            ]
        # MACs assigned by Docker should be assigned by Docker again.
        networks[name].pop("MacAddress", None)
    return {
        **config,
        "HostConfig": host,
        "NetworkingConfig": {"EndpointsConfig": networks},
    }


def assert_idle(engine, container):
    if container["State"].get("Paused"):
        raise RuntimeError("Unpause the application before updating.")
    if not container["State"]["Running"]:
        return
    status = engine.api(container["Id"], "/v1/status")
    busy = {"starting", "running", "stopping", "draining"}
    if (
        status.get("analysis_state") in busy
        or status.get("pipeline_state") in busy
        or status.get("processing_cleanup") == "closing"
    ):
        raise RuntimeError("Stop all analysis and wait for cleanup before updating.")
    if engine.api(container["Id"], "/v1/deployment/settings").get("autostart"):
        raise RuntimeError(
            "Disable automatic analysis startup in the console before updating."
        )


def wait_ready(engine, identity, timeout=150):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        container = engine.inspect(identity)
        if not container["State"]["Running"]:
            raise RuntimeError("Updated application stopped unexpectedly.")
        health = container["State"].get("Health", {}).get("Status")
        if health in (None, "healthy"):
            try:
                engine.api(identity, "/v1/ui/info")
                engine.api(identity, "/v1/status")
                return
            except RuntimeError:
                pass
        time.sleep(2)
    raise RuntimeError("Updated application did not become ready in time.")


def save_record(path, record):
    # Contains IDs and image names only, never environment variables or keys.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def tag_image(engine, image, reference):
    if reference.startswith("sha256:") or "@" in reference:
        return
    repository, separator, tag = reference.rpartition(":")
    if not separator or "/" in tag:
        repository, tag = reference, "latest"
    engine.request(
        "POST",
        f"/images/{image}/tag?repo={quote(repository, safe='')}&tag={quote(tag)}",
    )


def upgrade(
    engine,
    manifest,
    image,
    record_path,
    check_only=False,
    project="goodvision-harness",
    base_id=None,
    installation=None,
    container_names=None,
    verification_base_id=None,
):
    requested_names = set(container_names or ())
    if requested_names and (
        len(requested_names) != len(container_names)
        or any(
            not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name)
            for name in requested_names
        )
    ):
        raise RuntimeError("Invalid or duplicate container names.")
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if requested_names and requested_names != {
            item["name"] for item in record["containers"]
        }:
            raise RuntimeError("Containers differ from the recorded update.")
        if record.get("state") == "complete":
            for item in record["containers"]:
                if engine.inspect(item["name"])["Image"] != record["new_image_id"]:
                    raise RuntimeError("The installation changed since this update.")
            print("This update is already installed.", flush=True)
            return
    base_id = base_id or manifest["base_id"]
    targets = []
    for candidate in engine.request("GET", "/containers/json?all=1"):
        names = {name.lstrip("/") for name in candidate.get("Names", [])}
        if requested_names and not names & requested_names:
            continue
        labels = candidate.get("Labels") or {}
        if (
            labels.get("com.docker.compose.project") != project
            or labels.get("goodvision.update.helper") == "true"
            # Match the Windows selector: retained pre-update containers are
            # backups, not application lanes that should be upgraded.
            or any("-before-" in name for name in candidate.get("Names", []))
            or not re.fullmatch(
                r"harness(?:-\d+)?", labels.get("com.docker.compose.service", "")
            )
        ):
            continue
        if candidate["ImageID"] != base_id:
            raise RuntimeError(
                "Main application versions differ. No containers were changed."
            )
        targets.append(engine.inspect(candidate["Id"]))
    if (
        requested_names
        and {target["Name"].lstrip("/") for target in targets} != requested_names
    ):
        raise RuntimeError(
            "One or more selected application containers were not found."
        )
    if not targets:
        raise RuntimeError("No compatible GoodVision main application found.")
    verify_installation(engine, manifest, base_id, targets, verification_base_id)
    for target in targets:
        assert_idle(engine, target)
        replacement_config(target, image, continuous_playback_vlm=manifest.get("continuous_playback_vlm", False))
    metadata = read_metadata(installation, base_id)
    print(f"Verified {len(targets)} application containers.", flush=True)
    if check_only:
        return
    new_id = engine.request("GET", f"/images/{quote(image, safe='')}/json")["Id"]
    prepared = []
    # Create the replacements before interrupting the installed services.
    try:
        for target in targets:
            config = replacement_config(target, image, continuous_playback_vlm=manifest.get("continuous_playback_vlm", False))
            if "com.docker.compose.image" in (config.get("Labels") or {}):
                config["Labels"]["com.docker.compose.image"] = new_id
            name = target["Name"].lstrip("/")
            temporary = quote(name + "-update-" + manifest["release"])
            created = engine.request(
                "POST", f"/containers/create?name={temporary}", config
            )
            prepared.append(
                {
                    "name": name,
                    "old_id": target["Id"],
                    "new_id": created["Id"],
                    "running": target["State"]["Running"],
                }
            )
    except Exception:
        for item in prepared:
            engine.request("DELETE", f"/containers/{item['new_id']}?force=1")
        raise
    record = {
        "state": "updating",
        "release": manifest["release"],
        "base_id": base_id,
        "new_image_id": new_id,
        "containers": prepared,
    }
    save_record(record_path, record)
    for item in prepared:
        if item["running"]:
            engine.request("POST", f"/containers/{item['old_id']}/stop?t=30")
        # No volume deletion and no retained rollback containers.
        engine.request("DELETE", f"/containers/{item['old_id']}")
        engine.request(
            "POST", f"/containers/{item['new_id']}/rename?name={quote(item['name'])}"
        )
    for item in prepared:
        if item["running"]:
            engine.request("POST", f"/containers/{item['new_id']}/start")
    for item in prepared:
        if item["running"]:
            wait_ready(engine, item["new_id"])
    for reference in {target["Config"]["Image"] for target in targets}:
        tag_image(engine, new_id, reference)
    update_metadata(installation, metadata, manifest, new_id)
    record["state"] = "complete"
    save_record(record_path, record)
    print(
        "Update ready. Open the existing console URL. Analysis remains stopped.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--project", default="goodvision-harness")
    parser.add_argument("--base-id")
    parser.add_argument("--verification-base-id")
    parser.add_argument("--installation", type=Path)
    parser.add_argument("--containers", help="Comma-separated exact container names")
    parser.add_argument("--manifest", type=Path, default=Path("/update/manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    try:
        upgrade(
            Engine(),
            manifest,
            manifest["image"],
            Path("/state/update-state.json"),
            args.check_only,
            args.project,
            args.base_id,
            args.installation,
            args.containers.split(",") if args.containers else None,
            args.verification_base_id,
        )
    except Exception as error:
        print(f"Update stopped: {error}", flush=True)
        raise SystemExit(1) from None
