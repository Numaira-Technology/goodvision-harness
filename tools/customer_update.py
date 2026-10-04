"""Customer update packages: verified images, automatic local drain, no CLI setup."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .harness_update.release_metadata import (
        atomic_write,
        read_metadata,
        update_metadata,
    )
    from .update_docker import (
        LABEL,
        Docker,
        UpdateError,
        assert_idle,
        check_config,
        current_containers,
        discover,
        image_id,
        launcher_references,
        selected_service,
        update,
        update_lock,
    )
else:
    from harness_update.release_metadata import (
        atomic_write,
        read_metadata,
        update_metadata,
    )
    from update_docker import (
        LABEL,
        Docker,
        UpdateError,
        assert_idle,
        check_config,
        current_containers,
        discover,
        image_id,
        launcher_references,
        selected_service,
        update,
        update_lock,
    )

PRODUCT = "GoodVision Desktop Update"


class CustomerError(UpdateError):
    """A short message suitable for the update window."""


def load_package(directory: Path) -> dict:
    directory = Path(directory).resolve()
    try:
        manifest = json.loads((directory / "update-package.json").read_text("utf-8"))
        if (
            manifest["product"] != PRODUCT
            or manifest["schema_version"] != 1
            or manifest["component"] not in {"harness", "contour"}
            or not re.fullmatch(
                r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", manifest["release"]
            )
            or manifest["archive"]["file"] != "images.tar.gz"
            or not re.fullmatch(r"[a-f0-9]{64}", manifest["archive"]["sha256"])
        ):
            raise ValueError
        roles = (
            {"harness"} if manifest["component"] == "harness" else {"contour", "engine"}
        )
        if set(manifest["images"]) != roles:
            raise ValueError
        for image in manifest["images"].values():
            if (
                not isinstance(image["reference"], str)
                or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.:/@-]*", image["reference"]
                )
                or not re.fullmatch(r"sha256:[a-f0-9]{64}", image["id"])
            ):
                raise ValueError
        archive = directory / "images.tar.gz"
        if not archive.is_file() or archive.is_symlink():
            raise ValueError
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise CustomerError(
            "This update folder is incomplete. Extract the whole ZIP into a new folder."
        ) from error
    return manifest


def verify_archive(directory, manifest, progress):
    progress("Checking your update package...")
    digest = hashlib.sha256()
    with (Path(directory) / manifest["archive"]["file"]).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != manifest["archive"]["sha256"]:
        raise CustomerError(
            "The update file is damaged. Download a fresh copy and try again."
        )


def find_projects(component, *, docker=None):
    docker = docker or Docker()
    ids = docker.run("ps", "-aq", "--filter", "label=" + LABEL + "project").split()
    containers = docker.json("inspect", *ids) if ids else []
    projects = set()
    for container in containers:
        labels = container["Config"].get("Labels") or {}
        if (
            selected_service(component, labels.get(LABEL + "service", ""))
            and labels.get(LABEL + "oneoff", "False").lower() != "true"
            and labels.get("goodvision.update.helper") != "true"
            and "-before-" not in container["Name"]
        ):
            projects.add(labels[LABEL + "project"])
    return sorted(projects)


@contextmanager
def stopped_harness(docker, deployment, progress, timeout=150):
    """Only the autostart boolean is changed, never connection/key settings."""
    saved = {}
    try:
        progress("Pausing analysis safely...")
        # Stop every lane before changing settings, since settings may be locked
        # while another lane is draining. These calls stay inside the containers.
        for service, container in deployment.containers.items():
            if not container["State"].get("Running"):
                raise CustomerError("Open GoodVision first, then try the update again.")
            settings = docker.api(container["Id"], "harness", "/v1/deployment/settings")
            if type(settings.get("autostart")) is not bool:
                raise CustomerError(
                    "GoodVision did not respond. Open it and try again."
                )
            saved[service] = settings["autostart"]
            docker.api(container["Id"], "harness", "/v1/analysis/stop", {})
        deadline = time.monotonic() + timeout
        while True:
            finished = True
            for container in deployment.containers.values():
                status = docker.api(container["Id"], "harness", "/v1/status")
                if (
                    status.get("analysis_state")
                    in {"starting", "running", "stopping", "draining"}
                    or status.get("pipeline_state")
                    in {"starting", "running", "stopping", "draining"}
                    or status.get("processing_cleanup") == "closing"
                ):
                    finished = False
            if finished:
                break
            if time.monotonic() >= deadline:
                raise CustomerError(
                    "Analysis is still stopping. Wait a moment and try again."
                )
            time.sleep(2)
        for container in deployment.containers.values():
            docker.api(
                container["Id"],
                "harness",
                "/v1/deployment/settings",
                {"autostart": False},
                method="PUT",
            )
        yield
    finally:
        if saved:
            failure_pending = sys.exc_info()[0] is not None
            try:
                containers = current_containers(docker, deployment)
                for service, autostart in saved.items():
                    docker.api(
                        containers[service]["Id"],
                        "harness",
                        "/v1/deployment/settings",
                        {"autostart": autostart},
                        method="PUT",
                    )
            except UpdateError as error:
                if not failure_pending:
                    raise CustomerError(
                        "The update finished, but automatic-start settings could "
                        "not be restored. Contact support."
                    ) from error


def verified_launcher_metadata(docker, deployment, manifest):
    if (
        deployment.component != "harness"
        or not (deployment.directory / "RELEASE.json").is_file()
    ):
        return {}, None
    bases = {c["Image"] for c in deployment.containers.values()}
    if len(bases) != 1:
        raise CustomerError(
            "This installation needs a different update package. Contact support."
        )
    try:
        backup = read_metadata(deployment.directory, bases.pop())
    except RuntimeError as error:
        raise CustomerError(
            "Your installed files could not be verified. Contact support."
        ) from error
    sources = docker.json(
        "run",
        "--rm",
        "--network",
        "none",
        "--entrypoint",
        "python",
        manifest["images"]["harness"]["id"],
        "-c",
        "import hashlib,importlib.util,json,pathlib;"
        "root=pathlib.Path(importlib.util.find_spec('goodvision_harness').origin).parent;"
        "print(json.dumps({p.relative_to(root).as_posix():"
        "hashlib.sha256(p.read_bytes()).hexdigest() "
        "for p in root.rglob('*') if p.is_file() and not p.is_symlink() "
        "and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}))",
    )
    original = json.loads(base64.b64decode(backup["SOURCE_HASHES.json"]))["files"]
    # Keep every verified inventory entry. A package that removes an expected
    # file needs a separate release migration; never erase integrity checks.
    if (
        not original
        or not sources
        or not set(original) <= set(sources)
        or any(not re.fullmatch(r"[a-f0-9]{64}", digest) for digest in sources.values())
    ):
        raise CustomerError(
            "This update does not match your installed edition. Contact support."
        )
    source_manifest = {
        "release": manifest["release"],
        "files": [
            {"path": "payload/" + name, "sha256": digest}
            for name, digest in sources.items()
        ],
    }
    return backup, source_manifest


def install(directory, project=None, progress=lambda _message: None, *, docker=None):
    docker = docker or Docker()
    directory = Path(directory).resolve()
    manifest = load_package(directory)
    verify_archive(directory, manifest, progress)
    progress("Finding your GoodVision installation...")
    deployment = discover(docker, manifest["component"], project)
    check_config(docker, deployment)
    if manifest["component"] == "contour":
        # Other computers may own Contour sessions. Never delete their sessions
        # or force-stop analysis without contacting the owning main console.
        try:
            assert_idle(docker, deployment)
        except UpdateError as error:
            raise CustomerError(
                "Stop analysis in the main GoodVision console, then click Try again."
            ) from error
    with update_lock(deployment):
        progress("Preparing the new version. Please keep this window open...")
        docker.run(
            "image", "load", "-i", str(directory / "images.tar.gz"), timeout=7200
        )
        for image in manifest["images"].values():
            if image_id(docker, image["id"]) != image["id"]:
                raise CustomerError(
                    "This update contains the wrong images. Download a fresh copy."
                )
        images = {
            service: manifest["images"][
                "harness" if deployment.component == "harness" else service
            ]["id"]
            for service in deployment.services
        }
        launcher_references(docker, deployment, images)
        metadata, source_manifest = verified_launcher_metadata(
            docker, deployment, manifest
        )

        def apply():
            progress(
                "Installing the update. Your saved settings and models are kept..."
            )
            update(
                docker,
                deployment,
                images=images,
                persist_references=True,
                lock_held=True,
            )
            if metadata:
                try:
                    update_metadata(
                        deployment.directory,
                        metadata,
                        source_manifest,
                        images["harness"],
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    restored = discover(
                        docker,
                        deployment.component,
                        deployment.project,
                        deployment.env_files,
                    )
                    try:
                        update(docker, restored, rollback=True, lock_held=True)
                    finally:
                        for name, contents in metadata.items():
                            atomic_write(
                                deployment.directory / name, base64.b64decode(contents)
                            )
                    raise CustomerError(
                        "The update could not finish. Your previous version "
                        "was restored. Contact support."
                    ) from error

        if deployment.component == "harness":
            with stopped_harness(docker, deployment, progress):
                apply()
        else:
            apply()
    progress("Update complete. Open GoodVision and start analysis when ready.")
