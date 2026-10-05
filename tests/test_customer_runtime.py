"""Preserve historical installed containers without relying on old Compose files."""

from __future__ import annotations

import copy
import hashlib

import pytest

from tools import customer_runtime as runtime
from tools import update_docker as updater


def identity(value):
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


class RuntimeDocker:
    """Model native container operations, including retained rollback containers."""

    def __init__(
        self,
        directory,
        component="harness",
        project="goodvision-harness",
        names=None,
        references=None,
        services=None,
    ):
        self.directory = directory
        self.component = component
        self.project = project
        self.services = services or (
            ("harness", "harness-02")
            if component == "harness"
            else ("contour", "engine")
        )
        names = names or {
            service: f"{project}-{service}-1" for service in self.services
        }
        references = references or {
            service: "goodvision-harness:20260930-legacy-gpuhost1"
            for service in self.services
        }
        self.calls = []
        self.created = []
        self.fail_create = 0
        self.lose_create_response = False
        self.fail_start = 0
        self.fail_tag = 0
        self.fail_remove = set()
        self.images = {
            reference: identity("old-" + reference) for reference in references.values()
        }
        self.candidates = {
            service: identity(
                "candidate-" + ("harness" if component == "harness" else service)
            )
            for service in self.services
        }
        self.containers = {}
        self.accepting = True
        for index, service in enumerate(self.services):
            old_id = hashlib.sha256((project + service).encode()).hexdigest()
            self.containers[old_id] = {
                "Id": old_id,
                "Name": "/" + names[service],
                "Image": self.images[references[service]],
                "Config": {
                    "Image": references[service],
                    "Entrypoint": None,
                    "Cmd": ["goodvision-harness"]
                    if component == "harness"
                    else ["python", "app.py"],
                    "Env": [
                        "LOCAL_CAMERA_TARGET_FPS=15",
                        "SCENE_ENGINE_MODE=active",
                        "SCENE_INDEPENDENT_VLM=true",
                        "PRIVATE_KEY=runtime-private-value",
                    ],
                    "WorkingDir": "/app",
                    "User": "10001:10001",
                    "ExposedPorts": {"8080/tcp": {}},
                    "Volumes": {"/data": {}, "/models": {}},
                    "Labels": {
                        updater.LABEL + "project": project,
                        updater.LABEL + "service": service,
                        updater.LABEL + "config-hash": "historical-stale-hash",
                        "goodvision.ui.release": "20260930-legacy-gpuhost1",
                    },
                },
                "HostConfig": {
                    "Binds": [r"D:\Customer Data\config.yaml:/app/config.yaml:ro"],
                    "Mounts": [
                        {
                            "Type": "bind",
                            "Source": "/opt/customer-models",
                            "Target": "/models",
                            "ReadOnly": True,
                        }
                    ],
                    "PortBindings": {
                        "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "0"}]
                    },
                    "RestartPolicy": {"Name": "unless-stopped", "MaximumRetryCount": 0},
                    "DeviceRequests": [
                        {
                            "Driver": "nvidia",
                            "Count": 0,
                            "DeviceIDs": ["1"],
                            "Capabilities": [["gpu"]],
                            "Options": {},
                        }
                    ],
                    "ReadonlyRootfs": True,
                    "Tmpfs": {"/tmp": "size=256m"},
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges:true"],
                },
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": "/opt/customer-models",
                        "Destination": "/models",
                        "RW": False,
                    },
                    {
                        "Type": "volume",
                        "Name": "existing-anonymous-data-" + service,
                        "Destination": "/data",
                        "RW": True,
                    },
                ],
                "NetworkSettings": {
                    "Ports": {
                        "8080/tcp": [
                            {"HostIp": "127.0.0.1", "HostPort": str(48123 + index)}
                        ]
                    },
                    "Networks": {
                        "customer-network": {
                            "Aliases": [
                                service,
                                "custom-" + service,
                                old_id,
                                old_id[:12],
                            ],
                            "MacAddress": "02:42:ac:11:00:02",
                        }
                    },
                },
                "State": {
                    "Running": True,
                    "Paused": False,
                    "OOMKilled": False,
                    "Restarting": False,
                },
            }
        self.originals = copy.deepcopy(self.containers)
        self.images.update(
            {candidate: candidate for candidate in self.candidates.values()}
        )

    def deployment(self):
        containers = {
            value["Config"]["Labels"][updater.LABEL + "service"]: copy.deepcopy(value)
            for value in self.containers.values()
            if "-before-" not in value["Name"] and "-update-" not in value["Name"]
        }
        return updater.Deployment(
            self.component, self.project, self.directory, (), (), containers
        )

    def run(self, *args, **kwargs):
        self.calls.append(("run", args))
        if args[0] == "info":
            return "linux"
        if args[0] == "ps":
            filters = [
                arg.partition("project=")[2]
                for arg in args
                if "label=" + updater.LABEL + "project=" in arg
            ]
            return " ".join(
                c["Id"]
                for c in self.containers.values()
                if not filters
                or c["Config"]["Labels"].get(updater.LABEL + "project") == filters[0]
            )
        if args[:2] == ("compose", "version"):
            return "Docker Compose fixture"
        if args[:2] == ("image", "tag"):
            if self.fail_tag:
                self.fail_tag -= 1
                raise updater.UpdateError("Could not create the rollback image tag")
            self.images[args[3]] = args[2]
            return ""
        if args[0] == "stop":
            for container_id in args[1:]:
                if container_id in self.containers:
                    self.containers[container_id]["State"]["Running"] = False
            return ""
        if args[0] == "rename":
            name = "/" + args[2].lstrip("/")
            assert not any(
                c["Name"] == name and c["Id"] != args[1]
                for c in self.containers.values()
            )
            self.containers[args[1]]["Name"] = name
            return ""
        if args[0] == "start":
            for container_id in args[1:]:
                value = self.containers[container_id]
                if self.fail_start and value["Image"] in self.candidates.values():
                    self.fail_start -= 1
                    raise updater.UpdateError("Replacement could not start")
                value["State"]["Running"] = True
            return ""
        if args[0] == "rm":
            assert "-v" not in args and "--volumes" not in args
            for container_id in args[1:]:
                if container_id in self.fail_remove:
                    raise updater.UpdateError("The old container is still retained")
                if container_id in self.containers:
                    del self.containers[container_id]
            return ""
        raise AssertionError(args)

    def json(self, *args, **kwargs):
        self.calls.append(("json", args))
        if args[:2] == ("context", "inspect"):
            return [{"Endpoints": {"docker": {"Host": "unix:///var/run/docker.sock"}}}]
        if args[0] == "inspect":
            inspected = []
            for value in args[1:]:
                container = self.containers.get(value) or next(
                    (
                        candidate
                        for candidate in self.containers.values()
                        if candidate["Name"].lstrip("/") == value.lstrip("/")
                    ),
                    None,
                )
                if container is None:
                    raise updater.UpdateError("No such container")
                inspected.append(container)
            return copy.deepcopy(inspected)
        if args[:2] == ("image", "inspect"):
            reference = args[2]
            return [
                {
                    "Id": self.images.get(reference, reference),
                    "Os": "linux",
                    "Architecture": "amd64",
                }
            ]
        raise AssertionError(args)

    def api(self, container_id, component, path, body=None, **kwargs):
        self.calls.append(("api", (container_id, component, path, body)))
        assert self.containers[container_id]["State"]["Running"]
        if component == "harness":
            if path == "/v1/status":
                return {
                    "analysis_state": "idle",
                    "pipeline_state": "idle",
                    "processing_cleanup": "complete",
                }
            if path == "/v1/deployment/settings":
                return {"autostart": False}
            assert path == "/v1/ui/info"
            return {"version": "fixture"}
        if path == "/api/admission":
            self.accepting = body["accepting_sessions"]
        else:
            assert path == "/api/status"
        return {
            "accepting_sessions": self.accepting,
            "gpu_ready": True,
            "active_sessions": 0,
            "requests": {"in_flight": 0},
        }

    def create_container(self, config, name, helper_image):
        self.calls.append(("create", (copy.deepcopy(config), name, helper_image)))
        if self.fail_create and len(self.created) + 1 == self.fail_create:
            raise updater.UpdateError("Could not stage replacement")
        container_id = "new-" + str(len(self.created) + 1)
        value = {
            "Id": container_id,
            "Name": "/" + name.lstrip("/"),
            "Image": config["Image"],
            "Config": {
                key: copy.deepcopy(value)
                for key, value in config.items()
                if key not in {"HostConfig", "NetworkingConfig"}
            },
            "HostConfig": copy.deepcopy(config["HostConfig"]),
            "Mounts": [],
            "NetworkSettings": {
                "Ports": copy.deepcopy(config["HostConfig"].get("PortBindings", {})),
                "Networks": copy.deepcopy(
                    config["NetworkingConfig"]["EndpointsConfig"]
                ),
            },
            "State": {
                "Running": False,
                "Paused": False,
                "OOMKilled": False,
                "Restarting": False,
            },
        }
        self.created.append(copy.deepcopy(value))
        self.containers[container_id] = value
        if self.lose_create_response:
            self.lose_create_response = False
            raise updater.UpdateError("The helper response was lost")
        return container_id


@pytest.mark.parametrize(
    "project,names,reference",
    [
        (
            "goodvision-harness-deploy",
            (
                "goodvision-harness-deploy-harness-1",
                "goodvision-harness-deploy-harness-02-1",
            ),
            "goodvision-harness:20260920-diagnostics2",
        ),
        (
            "goodvision-harness",
            ("goodvision-harness-harness-1", "goodvision-harness-harness-02-1"),
            "goodvision-harness:20260930-legacy-gpuhost1",
        ),
        (
            "goodvision-harness-deploy",
            ("goodvision-live-latest", "goodvision-contour-camera-02"),
            "goodvision-harness:20260921-cloud-health7",
        ),
        (
            "gv-update-check-full",
            ("gv-update-check-full-harness", "gv-update-check-full-harness-2"),
            "goodvision-update-validation:full",
        ),
    ],
)
def test_historical_container_names_and_image_tags_need_no_compose_files(
    tmp_path, project, names, reference
):
    docker = RuntimeDocker(
        tmp_path,
        project=project,
        names=dict(zip(("harness", "harness-02"), names, strict=False)),
        references={service: reference for service in ("harness", "harness-02")},
    )
    deployment = updater.discover(
        docker, "harness", runtime=True, state_directory=tmp_path
    )
    assert deployment.project == project
    assert deployment.services == ["harness", "harness-02"]
    assert not deployment.files
    assert {c["Name"].lstrip("/") for c in deployment.containers.values()} == set(names)
    assert not any(
        args[0] == "compose" and "config" in args
        for kind, args in docker.calls
        if kind in {"run", "json"}
    )


@pytest.mark.parametrize(
    "project,gateway,engine",
    [
        (
            "goodvision-contour",
            "goodvision-contour:0.1.0",
            "goodvision-contour-runtime:0.1.0",
        ),
        (
            "contour",
            "goodvision-contour:20260926",
            "goodvision-contour-engine:20260926",
        ),
    ],
)
def test_archived_contour_deployments_use_their_original_projects_and_image_references(
    tmp_path, project, gateway, engine
):
    docker = RuntimeDocker(
        tmp_path, "contour", project, references={"contour": gateway, "engine": engine}
    )
    deployment = updater.discover(
        docker, "contour", runtime=True, state_directory=tmp_path
    )
    assert deployment.project == project
    assert deployment.services == ["contour", "engine"]
    assert not deployment.files
    with updater.update_lock(deployment):
        runtime.runtime_update(docker, deployment, docker.candidates)
    assert docker.images[gateway] == docker.candidates["contour"]
    assert docker.images[engine] == docker.candidates["engine"]
    assert docker.accepting is True


def test_runtime_clone_preserves_actual_settings_volumes_gpu_and_network_aliases(
    tmp_path,
):
    docker = RuntimeDocker(tmp_path)
    deployment = docker.deployment()
    with updater.update_lock(deployment):
        runtime.runtime_update(docker, deployment, docker.candidates)
    assert len(docker.created) == 2
    for created in docker.created:
        service = created["Config"]["Labels"][updater.LABEL + "service"]
        original = next(
            c
            for c in docker.originals.values()
            if c["Config"]["Labels"][updater.LABEL + "service"] == service
        )
        assert created["Config"]["Env"] == original["Config"]["Env"]
        assert created["Config"]["Cmd"] == original["Config"]["Cmd"]
        assert created["HostConfig"]["Binds"] == original["HostConfig"]["Binds"]
        assert (
            created["HostConfig"]["DeviceRequests"]
            == original["HostConfig"]["DeviceRequests"]
        )
        assert (
            created["HostConfig"]["RestartPolicy"]
            == original["HostConfig"]["RestartPolicy"]
        )
        assert (
            created["HostConfig"]["PortBindings"]
            == original["NetworkSettings"]["Ports"]
        )
        assert {
            "Type": "volume",
            "Source": "existing-anonymous-data-" + service,
            "Target": "/data",
            "ReadOnly": False,
        } in created["HostConfig"]["Mounts"]
        assert created["NetworkSettings"]["Networks"]["customer-network"][
            "Aliases"
        ] == [service, "custom-" + service]
        assert (
            "MacAddress"
            not in created["NetworkSettings"]["Networks"]["customer-network"]
        )
    assert not set(docker.originals) & set(docker.containers)
    assert {c["Name"] for c in docker.containers.values()} == {
        c["Name"] for c in docker.originals.values()
    }
    for path in deployment.state_dir.rglob("*.json"):
        contents = path.read_text(encoding="utf-8")
        assert "runtime-private-value" not in contents
        assert "PRIVATE_KEY" not in contents


def test_every_replacement_is_staged_and_every_old_service_stopped_before_new_start(
    tmp_path,
):
    docker = RuntimeDocker(
        tmp_path,
        "contour",
        references={
            "contour": "goodvision-contour:20260926",
            "engine": "goodvision-contour-engine:20260926",
        },
    )
    deployment = docker.deployment()
    with updater.update_lock(deployment):
        runtime.runtime_update(docker, deployment, docker.candidates)
    creation_indexes = [
        i for i, (kind, _) in enumerate(docker.calls) if kind == "create"
    ]
    stop_indexes = [
        i
        for i, (kind, args) in enumerate(docker.calls)
        if kind == "run" and args[0] == "stop"
    ]
    starts = [
        i
        for i, (kind, args) in enumerate(docker.calls)
        if kind == "run" and args[0] == "start"
    ]
    assert max(creation_indexes) < min(stop_indexes)
    assert max(stop_indexes) < min(starts)


def test_staging_failure_leaves_original_services_running_without_downtime(tmp_path):
    docker = RuntimeDocker(tmp_path)
    docker.fail_create = 2
    deployment = docker.deployment()
    with updater.update_lock(deployment), pytest.raises(updater.UpdateError):
        runtime.runtime_update(docker, deployment, docker.candidates)
    assert docker.containers == docker.originals
    assert not any(
        kind == "run" and args[0] in {"stop", "rename", "start"}
        for kind, args in docker.calls
    )


def test_lost_creation_response_removes_staging_container_without_downtime(tmp_path):
    docker = RuntimeDocker(tmp_path)
    docker.lose_create_response = True
    deployment = docker.deployment()
    with updater.update_lock(deployment), pytest.raises(updater.UpdateError):
        runtime.runtime_update(docker, deployment, docker.candidates)
    assert len(docker.created) == 1
    assert docker.containers == docker.originals
    assert not any(
        kind == "run" and args[0] in {"stop", "rename", "start"}
        for kind, args in docker.calls
    )


def test_start_failure_restores_original_containers_names_images_and_running_state(
    tmp_path,
):
    docker = RuntimeDocker(tmp_path)
    docker.fail_start = 1
    deployment = docker.deployment()
    with updater.update_lock(deployment), pytest.raises(updater.UpdateError):
        runtime.runtime_update(docker, deployment, docker.candidates)
    assert docker.containers == docker.originals
    for original in docker.originals.values():
        assert docker.images[original["Config"]["Image"]] == original["Image"]


def test_finish_callback_runs_before_old_containers_are_removed_and_failure_recovers(
    tmp_path,
):
    docker = RuntimeDocker(tmp_path)
    deployment = docker.deployment()
    finished = []

    def finish():
        finished.append(True)
        assert set(docker.originals) <= set(docker.containers)
        assert all(
            not docker.containers[old_id]["State"]["Running"]
            for old_id in docker.originals
        )
        assert all(
            c["State"]["Running"]
            for c in docker.containers.values()
            if c["Id"] not in docker.originals
        )
        raise RuntimeError("Launcher metadata could not be saved")

    with updater.update_lock(deployment), pytest.raises(updater.UpdateError):
        runtime.runtime_update(docker, deployment, docker.candidates, finish=finish)
    assert finished == [True]
    assert docker.containers == docker.originals


def test_same_runtime_update_is_idempotent_without_another_restart(tmp_path):
    docker = RuntimeDocker(tmp_path)
    deployment = docker.deployment()
    with updater.update_lock(deployment):
        runtime.runtime_update(docker, deployment, docker.candidates)
    creates = len(docker.created)
    stops = sum(kind == "run" and args[0] == "stop" for kind, args in docker.calls)
    with updater.update_lock(deployment):
        runtime.runtime_update(docker, docker.deployment(), docker.candidates)
    assert len(docker.created) == creates
    assert (
        sum(kind == "run" and args[0] == "stop" for kind, args in docker.calls) == stops
    )


def test_state_write_failure_restores_runtime_before_launcher_metadata_commits(
    tmp_path, monkeypatch
):
    docker = RuntimeDocker(tmp_path)
    deployment = docker.deployment()
    finished = []
    write_json = runtime.write_json

    def fail_ready_write(path, value):
        if value["status"] == "ready":
            raise OSError("The update state disk is full")
        write_json(path, value)

    monkeypatch.setattr(runtime, "write_json", fail_ready_write)
    with updater.update_lock(deployment), pytest.raises(updater.UpdateError):
        runtime.runtime_update(
            docker, deployment, docker.candidates, finish=lambda: finished.append(True)
        )
    assert docker.containers == docker.originals
    assert not finished


def test_shared_and_distinct_launcher_tags_keep_each_lane_reference_across_updates(
    tmp_path,
):
    references = {
        "harness": "goodvision-harness:tagA",
        "harness-02": "goodvision-harness:tagB",
        "harness-03": "goodvision-harness:tagB",
    }
    docker = RuntimeDocker(tmp_path, references=references, services=tuple(references))
    for release in ("first", "second"):
        docker.candidates = {service: identity(release) for service in docker.services}
        deployment = docker.deployment()
        with updater.update_lock(deployment):
            runtime.runtime_update(docker, deployment, docker.candidates)
        for service, container in docker.deployment().containers.items():
            assert (
                container["Config"]["Labels"]["goodvision.update.launcher_image"]
                == references[service]
            )
            assert docker.images[references[service]] == docker.candidates[service]


def test_own_stopped_backups_retained_after_cleanup_failure_allow_subsequent_updates(
    tmp_path,
):
    docker = RuntimeDocker(tmp_path)
    docker.fail_remove = set(docker.originals)
    for release in ("first", "second", "third"):
        docker.candidates = {service: identity(release) for service in docker.services}
        deployment = docker.deployment()
        with updater.update_lock(deployment):
            runtime.runtime_update(docker, deployment, docker.candidates)
        assert set(docker.originals) <= set(docker.containers)
        for old_id in docker.originals:
            assert not docker.containers[old_id]["State"]["Running"]
            assert "-before-" in docker.containers[old_id]["Name"]
        assert all(
            container["Image"] == identity(release)
            for container in docker.deployment().containers.values()
        )


def test_unrelated_stopped_backup_sharing_launcher_tag_is_rejected(tmp_path):
    docker = RuntimeDocker(tmp_path)
    unrelated = copy.deepcopy(next(iter(docker.originals.values())))
    unrelated["Id"] = "unrelated-backup"
    unrelated["Name"] += "-before-unrelated-update"
    unrelated["State"]["Running"] = False
    docker.containers[unrelated["Id"]] = unrelated
    deployment = docker.deployment()
    with (
        updater.update_lock(deployment),
        pytest.raises(updater.UpdateError, match="shares"),
    ):
        runtime.runtime_update(docker, deployment, docker.candidates)
    assert not docker.created
    assert docker.containers[unrelated["Id"]] == unrelated


def test_recorded_backup_cannot_share_launcher_tags_while_running(tmp_path):
    docker = RuntimeDocker(tmp_path)
    docker.fail_remove = set(docker.originals)
    deployment = docker.deployment()
    with updater.update_lock(deployment):
        runtime.runtime_update(docker, deployment, docker.candidates)
    old_id = next(iter(docker.originals))
    docker.containers[old_id]["State"]["Running"] = True
    docker.candidates = {service: identity("second") for service in docker.services}
    deployment = docker.deployment()
    created = len(docker.created)
    with (
        updater.update_lock(deployment),
        pytest.raises(updater.UpdateError, match="shares"),
    ):
        runtime.runtime_update(docker, deployment, docker.candidates)
    assert len(docker.created) == created


def test_contour_preswitch_failure_restores_admission_and_keeps_originals_running(
    tmp_path,
):
    docker = RuntimeDocker(
        tmp_path,
        "contour",
        references={
            "contour": "goodvision-contour:0.1.0",
            "engine": "goodvision-contour-runtime:0.1.0",
        },
    )
    docker.fail_tag = 1
    deployment = docker.deployment()
    with updater.update_lock(deployment), pytest.raises(updater.UpdateError):
        runtime.runtime_update(docker, deployment, docker.candidates)
    assert docker.accepting is True
    assert docker.containers == docker.originals
    assert not any(
        kind == "run" and args[0] in {"stop", "rename", "start"}
        for kind, args in docker.calls
    )


def test_staged_cleanup_failure_attempts_every_container_and_restores_admission(
    tmp_path,
):
    docker = RuntimeDocker(
        tmp_path,
        "contour",
        references={
            "contour": "goodvision-contour:0.1.0",
            "engine": "goodvision-contour-runtime:0.1.0",
        },
    )
    docker.fail_tag = 1
    docker.fail_remove = {"new-1"}
    deployment = docker.deployment()
    with updater.update_lock(deployment), pytest.raises(updater.UpdateError):
        runtime.runtime_update(docker, deployment, docker.candidates)
    assert docker.accepting is True
    assert {key: docker.containers[key] for key in docker.originals} == docker.originals
    assert not docker.containers["new-1"]["State"]["Running"]
    assert "new-2" not in docker.containers
    cleanup_ids = {
        args[-1]
        for kind, args in docker.calls
        if kind == "run" and args[:2] == ("rm", "-f")
    }
    assert cleanup_ids == {container["Id"] for container in docker.created}
