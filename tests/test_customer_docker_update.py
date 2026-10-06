"""Check customer packages and automatic preparation without a Docker daemon."""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import queue
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from test_docker_update import FakeDocker

from tools import customer_update as customer
from tools import customer_update_gui as gui
from tools import package_customer_docker_update as packager
from tools import update_docker as updater


class CustomerDocker(FakeDocker):
    def __init__(self, directory, component="harness"):
        super().__init__(directory, component)
        self.loaded_images = {}
        self.autostarts = {service: False for service in self.services}
        self.running_analysis = False
        self.fail_stop = False
        self.load_count = 0
        self.load_has_lock = False
        self.source_hashes = {}
        self.staged = {}
        self.backups = {}
        for service, container in self.containers.items():
            container["Config"]["Image"] = self.config["services"][service]["image"]
            container["HostConfig"] = {"Binds": ["private-data:/data"]}
            container["NetworkSettings"] = {"Ports": {}, "Networks": {}}
            container["Mounts"] = []

    def all_containers(self):
        return [
            *self.containers.values(),
            *self.staged.values(),
            *self.backups.values(),
        ]

    def create_container(self, config, name, helper_image):
        self.calls.append(("create", (name, helper_image)))
        self.generation += 1
        identity = f"created-{self.generation}"
        self.staged[identity] = {
            "Id": identity,
            "Name": "/" + name,
            "Image": config["Image"],
            "Config": copy.deepcopy(
                {
                    key: value
                    for key, value in config.items()
                    if key not in {"HostConfig", "NetworkingConfig"}
                }
            ),
            "HostConfig": copy.deepcopy(config["HostConfig"]),
            "NetworkSettings": {
                "Ports": {},
                "Networks": copy.deepcopy(
                    config["NetworkingConfig"]["EndpointsConfig"]
                ),
            },
            "Mounts": [],
            "State": {"Running": False, "Paused": False},
        }
        return identity

    def run(self, *args, **kwargs):
        if args[0] == "load" or args[:2] == ("image", "load"):
            self.calls.append(("run", args))
            self.images.update(self.loaded_images)
            self.load_count += 1
            self.load_has_lock = (self.deployment().state_dir / "update.lock").is_file()
            return "Loaded image fixture"
        if args[0] == "ps":
            self.calls.append(("run", args))
            project = args[-1].partition("project=")[2]
            return " ".join(
                c["Id"]
                for c in self.all_containers()
                if not project
                or c["Config"]["Labels"].get(updater.LABEL + "project") == project
            )
        if args[0] in {"stop", "start", "rename", "rm"}:
            self.calls.append(("run", args))
            if args[0] == "start" and self.fail_up:
                self.fail_up = False
                raise updater.UpdateError("Replacement startup failed")
            identities = args[3:] if args[0] == "stop" else args[1:]
            if args[0] == "rm":
                identities = [identity for identity in identities if identity != "-f"]
                for identity in identities:
                    for values in (self.containers, self.staged, self.backups):
                        for key, container in list(values.items()):
                            if container["Id"] == identity:
                                del values[key]
            elif args[0] == "rename":
                identity, name = args[1:]
                container = next(
                    c for c in self.all_containers() if c["Id"] == identity
                )
                for values in (self.containers, self.staged, self.backups):
                    for key, value in list(values.items()):
                        if value is container:
                            del values[key]
                container["Name"] = "/" + name
                if "-before-" in name:
                    self.backups[identity] = container
                else:
                    service = container["Config"]["Labels"][updater.LABEL + "service"]
                    self.containers[service] = container
            else:
                for identity in identities:
                    container = next(
                        c for c in self.all_containers() if c["Id"] == identity
                    )
                    container["State"]["Running"] = args[0] == "start"
            return ""
        return super().run(*args, **kwargs)

    def api(self, identity, component, path, body=None, method=None):
        if component != "harness":
            return super().api(identity, component, path, body)
        self.calls.append(("api", (identity, component, path, body, method)))
        container = next(c for c in self.all_containers() if c["Id"] == identity)
        service = container["Config"]["Labels"][updater.LABEL + "service"]
        if path == "/v1/deployment/settings":
            if body is not None:
                assert method == "PUT"
                assert set(body) == {"autostart"}
                assert (self.deployment().state_dir / "update.lock").is_file()
                self.autostarts[service] = body["autostart"]
            return {"autostart": self.autostarts[service]}
        if path == "/v1/analysis/stop":
            assert body == {}
            assert (self.deployment().state_dir / "update.lock").is_file()
            if self.fail_stop:
                raise updater.UpdateError("Stopping analysis failed")
            self.running_analysis = False
            self.harness_status["analysis_state"] = "idle"
            self.harness_status["pipeline_state"] = "idle"
            self.harness_status["processing_cleanup"] = "complete"
            return copy.deepcopy(self.harness_status)
        if path == "/v1/status":
            return copy.deepcopy(self.harness_status)
        assert path == "/v1/ui/info"
        return {"version": "fixture"}

    def json(self, *args, **kwargs):
        if args[0] == "inspect":
            self.calls.append(("json", args))
            return copy.deepcopy(
                [
                    c
                    for c in self.all_containers()
                    if c["Id"] in args[1:] or c["Name"].lstrip("/") in args[1:]
                ]
            )
        if args[0] == "run":
            self.calls.append(("json", args))
            assert args[1:5] == ("--rm", "--network", "none", "--entrypoint")
            return copy.deepcopy(self.source_hashes)
        if args[:2] == ("image", "inspect") and args[2].startswith("sha256:"):
            self.calls.append(("json", args))
            if args[2] not in self.images.values():
                raise updater.UpdateError("The expected image was not loaded")
            return [{"Id": args[2]}]
        return super().json(*args, **kwargs)


def make_package(directory, component="harness"):
    directory.mkdir()
    archive = directory / "images.tar.gz"
    archive.write_bytes(gzip.compress(b"fixture image archive"))
    roles = ("harness",) if component == "harness" else ("contour", "engine")
    manifest = {
        "product": "GoodVision Desktop Update",
        "schema_version": 1,
        "release": "test-v2",
        "component": component,
        "archive": {
            "file": archive.name,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        },
        "images": {
            role: {
                "reference": f"goodvision-{role}:test-v2",
                "id": "sha256:" + hashlib.sha256(role.encode()).hexdigest(),
            }
            for role in roles
        },
    }
    write_manifest(directory, manifest)
    return manifest


def write_manifest(directory, manifest):
    (directory / "update-package.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


@pytest.mark.parametrize("component", ["harness", "contour"])
def test_customer_manifest_validates_matching_complete_roles(tmp_path, component):
    directory = tmp_path / "customer update"
    manifest = make_package(directory, component)
    assert customer.load_package(directory) == manifest


@pytest.mark.parametrize("corruption", ["archive", "escape", "roles", "image-id", "config-id"])
def test_invalid_customer_package_never_loads_images(tmp_path, corruption):
    directory = tmp_path / "customer update"
    manifest = make_package(directory)
    if corruption == "archive":
        (directory / "images.tar.gz").write_bytes(b"changed archive")
    elif corruption == "escape":
        manifest["archive"]["file"] = "../outside.tar.gz"
    elif corruption == "roles":
        manifest["images"]["engine"] = manifest["images"]["harness"]
    elif corruption == "image-id":
        manifest["images"]["harness"]["id"] = "sha256:not-an-image-id"
    else:
        manifest["images"]["harness"]["config_id"] = "sha256:not-an-image-id"
    write_manifest(directory, manifest)
    installed = tmp_path / "installed"
    installed.mkdir()
    docker = CustomerDocker(installed)
    with pytest.raises((updater.UpdateError, ValueError)):
        customer.install(directory, docker=docker)
    assert docker.load_count == 0
    assert not docker.mutations


def test_loaded_image_id_must_match_manifest_before_analysis_stops(tmp_path):
    directory = tmp_path / "customer update"
    manifest = make_package(directory)
    installed = tmp_path / "installed"
    installed.mkdir()
    docker = CustomerDocker(installed)
    reference = manifest["images"]["harness"]["reference"]
    docker.loaded_images[reference] = "sha256:" + "0" * 64
    with pytest.raises(updater.UpdateError):
        customer.install(directory, docker=docker)
    assert docker.load_count == 1
    assert not any(
        kind == "api" and args[2] == "/v1/analysis/stop" for kind, args in docker.calls
    )
    assert not any("stop" in args or "up" in args for args in docker.mutations)


@pytest.mark.parametrize("storage", ["containerd", "classic"])
def test_customer_update_accepts_only_the_recorded_storage_identity(tmp_path, storage):
    directory = tmp_path / "package"
    manifest = make_package(directory)
    image = manifest["images"]["harness"]
    image["config_id"] = "sha256:" + "c" * 64
    write_manifest(directory, manifest)
    installed = tmp_path / "installed"
    installed.mkdir()
    docker = CustomerDocker(installed)
    actual = image["id"] if storage == "containerd" else image["config_id"]
    docker.loaded_images[image["reference"]] = actual
    customer.install(directory, docker=docker)
    assert all(docker.containers[role]["Image"] == actual for role in docker.services)


def test_unknown_storage_identity_is_refused_before_analysis_stops(tmp_path):
    directory = tmp_path / "package"
    manifest = make_package(directory)
    image = manifest["images"]["harness"]
    image["config_id"] = "sha256:" + "c" * 64
    write_manifest(directory, manifest)
    installed = tmp_path / "installed"
    installed.mkdir()
    docker = CustomerDocker(installed)
    docker.loaded_images[image["reference"]] = "sha256:" + "d" * 64
    with pytest.raises(customer.CustomerError, match="wrong images"):
        customer.install(directory, docker=docker)
    assert not any(kind == "create" for kind, _args in docker.calls)


def test_packager_records_config_id_from_the_exported_platform(tmp_path):
    config = json.dumps({"os": "linux", "architecture": "amd64"}).encode()
    config_id = "sha256:" + hashlib.sha256(config).hexdigest()
    platform = json.dumps({"config": {"digest": config_id}}).encode()
    platform_id = "sha256:" + hashlib.sha256(platform).hexdigest()
    index = json.dumps(
        {
            "manifests": [
                {
                    "digest": platform_id,
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
            ]
        }
    ).encode()
    identity = "sha256:" + hashlib.sha256(index).hexdigest()
    archive = tmp_path / "images.tar"
    with tarfile.open(archive, "w") as bundle:
        for digest, data in [
            (identity, index),
            (platform_id, platform),
            (config_id, config),
        ]:
            member = tarfile.TarInfo("blobs/sha256/" + digest[7:])
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    images = {"harness": {"id": identity, "reference": "prepared:v2"}}
    packager.add_config_ids(archive, images)
    assert images["harness"]["config_id"] == config_id


@pytest.mark.parametrize("failed_update", [False, True])
def test_customer_harness_stops_analysis_then_restores_autostart(
    tmp_path, failed_update
):
    directory = tmp_path / "customer update"
    manifest = make_package(directory)
    installed = tmp_path / "installed"
    installed.mkdir()
    docker = CustomerDocker(installed)
    docker.loaded_images = {
        image["reference"]: image["id"] for image in manifest["images"].values()
    }
    docker.running_analysis = True
    docker.harness_status.update(analysis_state="running", pipeline_state="running")
    docker.autostarts = {"harness": True, "harness-2": False}
    docker.fail_up = failed_update
    original_tags = {
        docker.config["services"][service]["image"]: docker.containers[service]["Image"]
        for service in docker.services
    }
    progress = []
    if failed_update:
        with pytest.raises(updater.UpdateError, match="previous images were restored"):
            customer.install(directory, docker=docker, progress=progress.append)
    else:
        customer.install(directory, docker=docker, progress=progress.append)
        expected = manifest["images"]["harness"]["id"]
        assert all(
            docker.containers[service]["Image"] == expected
            for service in docker.services
        )
    assert docker.autostarts == {"harness": True, "harness-2": False}
    assert docker.load_has_lock
    assert all(
        docker.images[reference]
        == (original if failed_update else manifest["images"]["harness"]["id"])
        for reference, original in original_tags.items()
    )
    assert docker.harness_status["analysis_state"] == "idle"
    assert progress
    calls = docker.calls
    stops = [
        i
        for i, (kind, args) in enumerate(calls)
        if kind == "api" and args[2] == "/v1/analysis/stop"
    ]
    replacement = next(
        i for i, (kind, args) in enumerate(calls) if kind == "run" and "stop" in args
    )
    assert stops and max(stops) < replacement
    assert not any("build" in args for kind, args in calls if kind == "run")
    if not failed_update:
        ready = max(
            i
            for i, (kind, args) in enumerate(calls)
            if kind == "api" and args[2] == "/v1/ui/info"
        )
        launcher_tags = [
            i
            for i, (kind, args) in enumerate(calls)
            if kind == "run"
            and args[:2] == ("image", "tag")
            and args[3] in original_tags
        ]
        assert launcher_tags and min(launcher_tags) > ready


def test_failed_analysis_stop_does_not_replace_and_restores_autostart(tmp_path):
    directory = tmp_path / "customer update"
    manifest = make_package(directory)
    installed = tmp_path / "installed"
    installed.mkdir()
    docker = CustomerDocker(installed)
    docker.loaded_images = {
        image["reference"]: image["id"] for image in manifest["images"].values()
    }
    docker.autostarts["harness"] = True
    docker.fail_stop = True
    with pytest.raises(updater.UpdateError, match="Stopping analysis failed"):
        customer.install(directory, docker=docker)
    assert docker.autostarts["harness"] is True
    assert not any("stop" in args or "up" in args for args in docker.mutations)


def test_frozen_customer_app_finds_package_next_to_executable(tmp_path, monkeypatch):
    executable = tmp_path / "folder with spaces" / "Update.exe"
    monkeypatch.setattr(gui.sys, "frozen", True, raising=False)
    monkeypatch.setattr(gui.sys, "executable", str(executable))
    monkeypatch.setattr(
        gui.sys, "_MEIPASS", str(tmp_path / "temporary runtime"), raising=False
    )
    assert gui.package_directory() == executable.parent
    assert gui.package_directory(tmp_path) == tmp_path


class Widget:
    def __init__(self):
        self.value = ""
        self.disabled = False
        self.options = {}
        self.hidden = False

    def set(self, value):
        self.value = value

    def get(self):
        return self.value

    def configure(self, **options):
        self.options.update(options)

    def state(self, values):
        self.disabled = "disabled" in values

    def pack(self, **options):
        self.hidden = False

    def pack_forget(self):
        self.hidden = True

    def start(self, *args):
        pass

    def stop(self):
        pass


class HeadlessRoot:
    def __init__(self):
        self.callbacks = []
        self.destroyed = False

    def after(self, interval, callback):
        self.callbacks.append((interval, callback))

    def destroy(self):
        self.destroyed = True


def headless_window(directory):
    window = gui.UpdateWindow.__new__(gui.UpdateWindow)
    window.root = HeadlessRoot()
    window.directory = directory
    window.events = queue.Queue()
    window.busy = False
    window.projects = []
    for name in (
        "status",
        "detail",
        "close_button",
        "action_button",
        "selection",
        "selection_frame",
        "progress",
    ):
        setattr(window, name, Widget())
    return window


def test_one_installation_starts_without_customer_selecting_paths(tmp_path):
    window = headless_window(tmp_path)
    selected = []
    window.start_install = selected.append
    window.show_projects(["installed"])
    assert selected == ["installed"]


def test_multiple_installations_require_selection_before_update(tmp_path):
    window = headless_window(tmp_path)
    selected = []
    window.start_install = selected.append
    window.show_projects(["installed", "second"])
    assert not selected
    assert window.action_button.disabled
    window.selection.set("second")
    window.selection_changed()
    assert not window.action_button.disabled
    window.start_selected()
    assert selected == ["second"]


def test_worker_progress_only_updates_ui_when_queue_is_polled(tmp_path):
    window = headless_window(tmp_path)
    window.status.set("Previous message")
    window.set_busy(True)
    window.progress_message("Installing the update")
    assert window.status.get() == "Previous message"
    window.events.put(("complete", None))
    window.poll_events()
    assert not window.busy
    assert "complete" in window.status.get().lower()
    assert window.action_button.hidden
    assert window.root.callbacks


def test_customer_cannot_close_window_during_installation(tmp_path):
    window = headless_window(tmp_path)
    window.set_busy(True)
    window.close()
    assert not window.root.destroyed
    window.set_busy(False)
    window.close()
    assert window.root.destroyed


def test_unexpected_error_details_and_credentials_stay_out_of_customer_ui():
    error = RuntimeError("Docker failed: PRIVATE_KEY=keep-private-key")
    message = gui.friendly_error(error)
    assert "PRIVATE_KEY" not in message
    assert "keep-private-key" not in message
    safe = customer.CustomerError("Stop analysis in the main GoodVision console.")
    assert gui.friendly_error(safe) == str(safe)


class ExportDocker:
    def __init__(self, component):
        roles = ("harness",) if component == "harness" else ("contour", "engine")
        self.configs = {
            role: json.dumps(
                {"os": "linux", "architecture": "amd64", "role": role}
            ).encode()
            for role in roles
        }
        self.images = {
            f"prepared-{role}:latest": {
                "Id": "sha256:" + hashlib.sha256(self.configs[role]).hexdigest(),
                "Os": "linux",
                "Architecture": "amd64",
            }
            for role in roles
        }
        self.roles = {role: f"prepared-{role}:latest" for role in roles}
        self.calls = []
        self.fail_save = False

    def json(self, *args, **kwargs):
        self.calls.append(args)
        assert args[:2] == ("image", "inspect")
        return [copy.deepcopy(self.images[args[2]])]

    def run(self, *args, **kwargs):
        self.calls.append(args)
        assert args[:3] == ("image", "save", "--output")
        if self.fail_save:
            raise updater.UpdateError("Image export failed")
        assert all(reference.startswith("sha256:") for reference in args[4:])
        with tarfile.open(args[3], "w") as archive:
            for role, data in self.configs.items():
                name = self.images[self.roles[role]]["Id"][7:] + ".json"
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        return ""


@pytest.mark.parametrize("component", ["harness", "contour"])
def test_shipped_zip_contains_executable_verified_images_and_simple_instructions(
    tmp_path, component
):
    docker = ExportDocker(component)
    executable = tmp_path / "prepared updater.exe"
    executable.write_bytes(b"MZfixture frozen updater")
    output = tmp_path / f"customer-{component}.zip"
    report = packager.package(
        output, "test-v2", component, docker.roles, executable, docker=docker
    )
    assert report["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["bytes"] == output.stat().st_size
    assert (
        output.with_suffix(".zip.sha256")
        .read_text(encoding="utf-8")
        .startswith(report["sha256"])
    )
    prefix = f"GoodVision-{component}-update-test-v2/"
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            prefix + name
            for name in (
                "Update.exe",
                "images.tar.gz",
                "update-package.json",
                "Read me.txt",
            )
        }
        assert archive.read(prefix + "Update.exe") == executable.read_bytes()
        manifest = json.loads(archive.read(prefix + "update-package.json"))
        image_archive = archive.read(prefix + "images.tar.gz")
        assert (
            hashlib.sha256(image_archive).hexdigest() == manifest["archive"]["sha256"]
        )
        assert set(manifest["images"]) == set(docker.roles)
        contents = gzip.decompress(image_archive).decode()
        for metadata in manifest["images"].values():
            assert metadata["id"][7:] in contents
            assert metadata["config_id"] == metadata["id"]
            assert metadata["reference"] not in contents
        instructions = archive.read(prefix + "Read me.txt").decode()
        assert "Double-click Update.exe" in instructions
        assert "Python" not in instructions
        assert "--" not in instructions
        extraction = tmp_path / "customer extracted"
        archive.extractall(extraction)
    folder = extraction / prefix.rstrip("/")
    assert customer.load_package(folder) == manifest
    customer.verify_archive(folder, manifest, lambda message: None)
    assert not output.with_suffix(".partial").exists()


def test_package_export_failure_removes_partial_release(tmp_path):
    docker = ExportDocker("harness")
    docker.fail_save = True
    executable = tmp_path / "Update.exe"
    executable.write_bytes(b"MZfixture")
    output = tmp_path / "new-release.zip"
    with pytest.raises(updater.UpdateError, match="export failed"):
        packager.package(
            output, "test-v2", "harness", docker.roles, executable, docker=docker
        )
    assert not output.exists()
    assert not output.with_suffix(".partial").exists()
    assert not list(tmp_path.glob("goodvision-update-*"))


@pytest.mark.parametrize("invalid", ["script", "architecture", "missing-role"])
def test_packager_rejects_incomplete_or_unsupported_release_before_export(
    tmp_path, invalid
):
    docker = ExportDocker("contour")
    executable = tmp_path / "Update.exe"
    executable.write_bytes(
        b"MZfixture" if invalid != "script" else b"python updater.py"
    )
    images = dict(docker.roles)
    if invalid == "architecture":
        docker.images[images["engine"]]["Architecture"] = "arm64"
    elif invalid == "missing-role":
        del images["engine"]
    output = tmp_path / "release.zip"
    with pytest.raises(ValueError):
        packager.package(
            output, "test-v2", "contour", images, executable, docker=docker
        )
    assert not any(call[:2] == ("image", "save") for call in docker.calls)
    assert not output.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows executable build")
def test_windowed_single_executable_build_bundles_customer_entrypoint(
    tmp_path, monkeypatch
):
    commands = []

    def freeze(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--distpath") + 1]) / "Update.exe"
        output.write_bytes(b"MZfrozen updater")

    monkeypatch.setattr(packager.subprocess, "run", freeze)
    assert packager.build_executable(tmp_path) == tmp_path / "Update.exe"
    command = commands[0]
    assert "--onefile" in command
    assert "--windowed" in command
    assert command[-1] == str(packager.ROOT / "customer_update_gui.py")
    assert command[command.index("--paths") + 1] == str(packager.ROOT)


def test_contour_customer_package_rejects_missing_engine_before_docker_load(tmp_path):
    directory = tmp_path / "customer update"
    manifest = make_package(directory, "contour")
    del manifest["images"]["engine"]
    write_manifest(directory, manifest)
    installed = tmp_path / "installed"
    installed.mkdir()
    docker = CustomerDocker(installed, "contour")
    with pytest.raises(customer.CustomerError):
        customer.install(directory, docker=docker)
    assert docker.load_count == 0
    assert not docker.mutations


def test_busy_contour_customer_install_does_not_interrupt_owning_harness(tmp_path):
    directory = tmp_path / "customer update"
    make_package(directory, "contour")
    installed = tmp_path / "installed"
    installed.mkdir()
    docker = CustomerDocker(installed, "contour")
    docker.contour_status["active_sessions"] = 1
    with pytest.raises(customer.CustomerError, match="main GoodVision console"):
        customer.install(directory, docker=docker)
    assert docker.load_count == 0
    assert not docker.mutations


def test_multiple_customer_installations_refuse_unselected_update(tmp_path):
    directory = tmp_path / "customer update"
    make_package(directory)
    installed = tmp_path / "installed"
    installed.mkdir()
    docker = CustomerDocker(installed)
    docker.containers["second"] = docker.container("harness", "another")
    assert customer.find_projects("harness", docker=docker) == ["another", "installed"]
    with pytest.raises(updater.UpdateError, match="--project"):
        customer.install(directory, docker=docker)
    assert docker.load_count == 0
    assert not docker.mutations


def installed_package(tmp_path):
    directory = tmp_path / "customer update"
    manifest = make_package(directory)
    installed = tmp_path / "installed"
    installed.mkdir()
    docker = CustomerDocker(installed)
    docker.loaded_images = {
        image["reference"]: image["id"] for image in manifest["images"].values()
    }
    return directory, manifest, docker


def test_customer_lock_is_acquired_before_loading_or_stopping_analysis(tmp_path):
    directory, _manifest, docker = installed_package(tmp_path)
    state = docker.deployment().state_dir
    state.mkdir(parents=True)
    lock = state / "update.lock"
    lock.write_text("another updater", encoding="utf-8")
    with pytest.raises(updater.UpdateError, match="update lock exists"):
        customer.install(directory, docker=docker)
    assert docker.load_count == 0
    assert not docker.mutations
    assert lock.read_text(encoding="utf-8") == "another updater"


def test_shared_launcher_tag_is_rejected_before_analysis_is_stopped(tmp_path):
    directory, _manifest, docker = installed_package(tmp_path)
    docker.containers["unrelated-worker"]["Config"]["Image"] = docker.config[
        "services"
    ]["harness"]["image"]
    with pytest.raises(updater.UpdateError, match="shares these launcher image tags"):
        customer.install(directory, docker=docker)
    assert not any(
        kind == "api" and args[2] == "/v1/analysis/stop" for kind, args in docker.calls
    )
    assert not any("stop" in args or "up" in args for args in docker.mutations)


def write_legacy_metadata(docker, source_hashes=None):
    identity = "sha256:" + "1" * 64
    for service in docker.services:
        docker.containers[service]["Image"] = identity
        reference = docker.config["services"][service]["image"]
        docker.images[reference] = identity
    source_hashes = (
        source_hashes
        if source_hashes is not None
        else {
            "service.py": "a" * 64,
            "config.yaml": "b" * 64,
        }
    )
    source = {
        "image_id": identity,
        "source": "original verified",
        "files": source_hashes,
    }
    release = {
        "release": "original",
        "accepted_image_ids": [identity],
        "edition": "keep",
    }
    files = {}
    for name, value in (("RELEASE.json", release), ("SOURCE_HASHES.json", source)):
        contents = (json.dumps(value, indent=2) + "\n").encode()
        files[name] = contents
        (docker.directory / name).write_bytes(contents)
    inventory = {
        "files": [
            {
                "path": name,
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
            for name, contents in files.items()
        ]
        + [{"path": "START.cmd", "bytes": 99, "sha256": "c" * 64}]
    }
    files["FILES.json"] = (json.dumps(inventory, indent=2) + "\n").encode()
    (docker.directory / "FILES.json").write_bytes(files["FILES.json"])
    docker.source_hashes = {
        "service.py": "d" * 64,
        "config.yaml": "e" * 64,
        "added.py": "f" * 64,
    }
    return files, identity


def test_verified_launcher_metadata_keeps_full_nonempty_source_inventory(tmp_path):
    directory, manifest, docker = installed_package(tmp_path)
    originals, old_identity = write_legacy_metadata(docker)
    customer.install(directory, docker=docker)
    new_identity = manifest["images"]["harness"]["id"]
    source = json.loads((docker.directory / "SOURCE_HASHES.json").read_bytes())
    assert source["files"] == docker.source_hashes
    assert source["files"]
    assert set(json.loads(originals["SOURCE_HASHES.json"])["files"]) <= set(
        source["files"]
    )
    release = json.loads((docker.directory / "RELEASE.json").read_bytes())
    assert set(release["accepted_image_ids"]) == {old_identity, new_identity}
    assert release["edition"] == "keep"
    inventory = json.loads((docker.directory / "FILES.json").read_bytes())
    assert next(item for item in inventory["files"] if item["path"] == "START.cmd") == {
        "path": "START.cmd",
        "bytes": 99,
        "sha256": "c" * 64,
    }
    assert customer.read_metadata(docker.directory, new_identity)


@pytest.mark.parametrize(
    "invalid", ["tampered", "missing-source", "empty-source", "empty-original"]
)
def test_unverifiable_legacy_source_inventory_refuses_before_drain(tmp_path, invalid):
    directory, _manifest, docker = installed_package(tmp_path)
    write_legacy_metadata(docker, {} if invalid == "empty-original" else None)
    if invalid == "tampered":
        (docker.directory / "SOURCE_HASHES.json").write_text(
            '{"files": {}}', encoding="utf-8"
        )
    elif invalid == "missing-source":
        del docker.source_hashes["config.yaml"]
    elif invalid == "empty-source":
        docker.source_hashes.clear()
    metadata_before = {
        name: (docker.directory / name).read_bytes()
        for name in ("RELEASE.json", "SOURCE_HASHES.json", "FILES.json")
    }
    with pytest.raises(customer.CustomerError):
        customer.install(directory, docker=docker)
    assert not any(
        kind == "api" and args[2] == "/v1/analysis/stop" for kind, args in docker.calls
    )
    assert not any("stop" in args or "up" in args for args in docker.mutations)
    assert {
        name: (docker.directory / name).read_bytes() for name in metadata_before
    } == metadata_before


def test_metadata_write_failure_restores_images_launcher_tags_and_exact_bytes(
    tmp_path, monkeypatch
):
    directory, _manifest, docker = installed_package(tmp_path)
    originals, identity = write_legacy_metadata(docker)

    def fail_metadata(root, *args):
        (root / "SOURCE_HASHES.json").write_bytes(b"partial failed metadata write")
        raise RuntimeError("Metadata disk write failed")

    monkeypatch.setattr(customer, "update_metadata", fail_metadata)
    with pytest.raises(updater.UpdateError, match="previous images were restored"):
        customer.install(directory, docker=docker)
    assert all(
        docker.containers[service]["Image"] == identity for service in docker.services
    )
    assert all(
        docker.images[docker.config["services"][service]["image"]] == identity
        for service in docker.services
    )
    assert {
        name: (docker.directory / name).read_bytes() for name in originals
    } == originals
    assert customer.read_metadata(docker.directory, identity)


def test_already_installed_image_still_promotes_original_launcher_tags(tmp_path):
    directory, manifest, docker = installed_package(tmp_path)
    identity = manifest["images"]["harness"]["id"]
    for service in docker.services:
        docker.containers[service]["Image"] = identity
    customer.install(directory, docker=docker)
    assert all(
        docker.images[docker.config["services"][service]["image"]] == identity
        for service in docker.services
    )


@pytest.mark.parametrize("status", [404, 401, 500])
def test_legacy_stop_fallback_only_for_missing_endpoint(tmp_path, status):
    directory, _manifest, docker = installed_package(tmp_path)
    original_api = docker.api
    legacy_stops = []

    def api(identity, component, path, body=None, method=None):
        if path == "/v1/analysis/stop":
            raise updater.ApiError(status)
        if path == "/v1/sessions/stop":
            legacy_stops.append(identity)
            assert body == {}
            return {"stopped": True}
        return original_api(identity, component, path, body, method)

    docker.api = api
    if status == 404:
        customer.install(directory, docker=docker)
        assert len(legacy_stops) == len(docker.services)
    else:
        with pytest.raises(updater.ApiError):
            customer.install(directory, docker=docker)
        assert not legacy_stops
        assert not any(kind == "create" for kind, _args in docker.calls)


def test_runtime_helper_passes_private_configuration_only_through_stdin(monkeypatch):
    from types import SimpleNamespace

    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout='{"Id":"created-id"}')

    monkeypatch.setattr(updater.subprocess, "run", run)
    config = {"Env": ["PRIVATE_KEY=never-show-this-key"], "Image": "sha256:verified"}
    assert (
        updater.Docker().create_container(config, "custom-name", "sha256:verified")
        == "created-id"
    )
    args, options = calls[0]
    assert "never-show-this-key" not in " ".join(args)
    assert json.loads(options["input"]) == {"name": "custom-name", "config": config}
    assert "--network" in args and "none" in args
    assert "goodvision.update.helper=true" in args
    assert "/var/run/docker.sock:/var/run/docker.sock" in args
    compile(args[-1], "<runtime-helper>", "exec")


def test_http_error_transport_keeps_endpoint_status_without_response_body():
    class HttpErrorDocker(updater.Docker):
        def json(self, *args, **kwargs):
            compile(args[-1], "<application-api>", "exec")
            return {"goodvision_update_http_error": 404}

    with pytest.raises(updater.ApiError) as failure:
        HttpErrorDocker().api("installed", "harness", "/v1/analysis/stop", {})
    assert failure.value.status == 404
    assert "404" in str(failure.value)


@pytest.mark.parametrize(
    "maximum,minimum,selected",
    [
        ("1.41", "1.12", "1.41"),
        ("1.44", "1.24", "1.44"),
        ("1.55", "1.44", "1.45"),
        ("1.40", "1.12", None),
        ("1.55", "1.46", None),
    ],
)
def test_runtime_helper_negotiates_daemon_api(
    monkeypatch, capsys, maximum, minimum, selected
):
    import http.client
    import io
    from types import SimpleNamespace

    calls = []

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args):
            calls.append(args)

        def getresponse(self):
            value = (
                {"ApiVersion": maximum, "MinAPIVersion": minimum}
                if len(calls) == 1
                else {"Id": "created-id"}
            )
            return SimpleNamespace(status=200, read=lambda: json.dumps(value).encode())

    code = []

    def run(*args, **kwargs):
        code.append(args[-1])
        return {"Id": "unused"}

    docker = updater.Docker()
    monkeypatch.setattr(docker, "json", run)
    docker.create_container({"Image": "sha256:verified"}, "custom name", "helper")
    monkeypatch.setattr(http.client, "HTTPConnection", Connection)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps({"name": "custom name", "config": {"Image": "sha256:verified"}})
        ),
    )
    if selected is None:
        with pytest.raises(SystemExit) as error:
            exec(code[0], {})
        assert error.value.code == 1
        assert calls == [("GET", "/version")]
    else:
        exec(code[0], {})
        assert calls[0] == ("GET", "/version")
        assert calls[1][0:2] == (
            "POST",
            f"/v{selected}/containers/create?name=custom%20name",
        )
        assert json.loads(calls[1][2]) == {"Image": "sha256:verified"}
        assert json.loads(capsys.readouterr().out) == {"Id": "created-id"}


def test_autostart_restore_attempts_every_lane_after_one_lane_fails(tmp_path):
    directory, _manifest, docker = installed_package(tmp_path)
    docker.autostarts = {"harness": True, "harness-2": True}
    original_api = docker.api
    restore_attempts = []

    def api(identity, component, path, body=None, method=None):
        if path == "/v1/deployment/settings" and body == {"autostart": True}:
            restore_attempts.append(identity)
            service = next(
                c["Config"]["Labels"][updater.LABEL + "service"]
                for c in docker.all_containers()
                if c["Id"] == identity
            )
            if service == "harness":
                raise updater.UpdateError("The first lane is temporarily unavailable")
        return original_api(identity, component, path, body, method)

    docker.api = api
    with pytest.raises(customer.CustomerError, match="Automatic-start settings"):
        customer.install(directory, docker=docker)
    assert len(restore_attempts) == 2
    assert docker.autostarts["harness-2"] is True


def test_interrupted_launcher_metadata_write_restores_exact_original_bytes(
    tmp_path, monkeypatch
):
    directory, _manifest, docker = installed_package(tmp_path)
    originals, identity = write_legacy_metadata(docker)

    def interrupted_metadata(root, *args):
        (root / "SOURCE_HASHES.json").write_bytes(b"interrupted partial write")
        raise KeyboardInterrupt

    monkeypatch.setattr(customer, "update_metadata", interrupted_metadata)
    with pytest.raises(updater.UpdateError, match="previous images were restored"):
        customer.install(directory, docker=docker)
    assert all(docker.containers[s]["Image"] == identity for s in docker.services)
    assert {
        name: (docker.directory / name).read_bytes() for name in originals
    } == originals


def test_known_windows_release_requires_original_launcher_metadata(tmp_path):
    directory, _manifest, docker = installed_package(tmp_path)
    for service in docker.services:
        docker.containers[service]["Image"] = customer.WINDOWS_BASE_IDS[0]
    with pytest.raises(customer.CustomerError, match="original installation folder"):
        customer.install(directory, docker=docker)
    assert not any(kind == "create" for kind, _args in docker.calls)
    assert not any(
        kind == "api" and args[2] == "/v1/analysis/stop" for kind, args in docker.calls
    )


def test_label_only_installation_uses_stable_lock_across_extracted_packages(
    tmp_path, monkeypatch
):
    directory, manifest, docker = installed_package(tmp_path)
    second = tmp_path / "second update folder"
    make_package(second)
    state = tmp_path / "local app data"
    monkeypatch.setenv("LOCALAPPDATA", str(state))
    for service in docker.services:
        labels = docker.containers[service]["Config"]["Labels"]
        for suffix in (
            "project.working_dir",
            "project.config_files",
            "project.environment_file",
        ):
            labels.pop(updater.LABEL + suffix)
    # The installed labels carry no folder; both package locations must resolve
    # to one host-level project lock, including while another update is loading.
    root = customer.runtime_state_directory()
    deployment = updater.discover(docker, "harness", runtime=True, state_directory=root)
    deployment.state_dir.mkdir(parents=True)
    (deployment.state_dir / "update.lock").write_text("another update")
    for package in (directory, second):
        with pytest.raises(updater.UpdateError, match="update lock exists"):
            customer.install(package, docker=docker)
    assert docker.load_count == 0
    assert manifest["component"] == "harness"


def test_updated_other_installation_launcher_label_prevents_shared_tag_update(tmp_path):
    directory, _manifest, docker = installed_package(tmp_path)
    worker = docker.containers["unrelated-worker"]
    worker["Config"]["Image"] = "sha256:another-updated-installation"
    worker["Config"]["Labels"]["goodvision.update.launcher_image"] = docker.config[
        "services"
    ]["harness"]["image"]
    with pytest.raises(updater.UpdateError, match="shares these launcher image tags"):
        customer.install(directory, docker=docker)
    assert not any(kind == "create" for kind, _args in docker.calls)


def test_same_image_retry_restores_launcher_tags_after_partial_promotion_failure(
    tmp_path,
):
    directory, manifest, docker = installed_package(tmp_path)
    candidate = manifest["images"]["harness"]["id"]
    before = {
        docker.config["services"][s]["image"]: docker.images[
            docker.config["services"][s]["image"]
        ]
        for s in docker.services
    }
    for service in docker.services:
        docker.containers[service]["Image"] = candidate
    run = docker.run
    promotions = []

    def failing_tag(*args, **kwargs):
        if args[:3] == ("image", "tag", candidate):
            promotions.append(args[3])
            if len(promotions) == 2:
                raise updater.UpdateError("The second tag promotion failed")
        return run(*args, **kwargs)

    docker.run = failing_tag
    with pytest.raises(updater.UpdateError, match="previous images were restored"):
        customer.install(directory, docker=docker)
    assert all(docker.images[reference] == old for reference, old in before.items())
    assert all(docker.containers[s]["Image"] == candidate for s in docker.services)
    assert not any(kind == "create" for kind, _args in docker.calls)


@pytest.mark.parametrize(
    "error,expected,retry",
    [
        ("Recovery did not become ready", "could not finish or restore", False),
        (
            "Update failed; previous images were restored",
            "previous version was restored",
            True,
        ),
        ("Could not restore Contour admission", "before you start analysis", False),
    ],
)
def test_customer_error_distinguishes_successful_and_failed_recovery(
    error, expected, retry
):
    message = gui.friendly_error(updater.UpdateError(error))
    assert expected in message
    assert ("Retry" in message) is retry
