"""Verify explicit Docker name lookup without selecting or changing other workloads."""

from __future__ import annotations

import copy
import hashlib

import pytest
from test_customer_docker_update import make_package
from test_customer_runtime import RuntimeDocker, identity

from tools import customer_discovery as discovery
from tools import customer_runtime as runtime
from tools import customer_update as customer
from tools import update_docker as updater


class ManualDocker(RuntimeDocker):
    """Expose role-specific APIs and gateway DNS alongside native inspection."""

    def __init__(self, directory, component="harness", *, labeled=False, services=None):
        services = services or (("harness",) if component == "harness" else None)
        references = (
            {s: "customer-main:installed" for s in services}
            if component == "harness"
            else {
                "contour": "customer-contour:installed",
                "engine": "customer-engine:installed",
            }
        )
        super().__init__(directory, component, services=services, references=references)
        self.roles = {}
        self.responses = {}
        self.dns = {}
        for index, container in enumerate(self.containers.values(), 2):
            role = container["Config"]["Labels"][updater.LABEL + "service"]
            self.roles[container["Id"]] = (
                "harness" if role.startswith("harness") else role
            )
            if not labeled:
                container["Config"]["Labels"] = {}
            container["NetworkSettings"]["Networks"]["customer-network"][
                "IPAddress"
            ] = f"172.20.0.{index}"
        if component == "contour":
            gateway = self.container_for("contour")
            engine = self.container_for("engine")
            gateway["Config"]["Env"].append("CONTOUR_ENGINE_URL=http://engine:8092")
            self.dns[gateway["Id"]] = [
                engine["NetworkSettings"]["Networks"]["customer-network"]["IPAddress"]
            ]

    def container_for(self, role):
        return next(c for c in self.containers.values() if self.roles[c["Id"]] == role)

    def add_container(self, name, role, *, reference="customer-main:installed"):
        container = copy.deepcopy(next(iter(self.containers.values())))
        container_id = hashlib.sha256(name.encode()).hexdigest()
        container.update(
            Id=container_id, Name="/" + name, Image=identity("old-" + reference)
        )
        container["Config"]["Image"] = reference
        container["Config"]["Labels"] = {}
        container["NetworkSettings"]["Networks"]["customer-network"]["IPAddress"] = (
            f"172.20.0.{len(self.containers) + 2}"
        )
        self.containers[container_id] = container
        self.roles[container_id] = role
        self.images[reference] = container["Image"]
        return container

    def json(self, *args, **kwargs):
        if args[0] == "exec":
            self.calls.append(("json", args))
            if args[1] not in self.dns:
                raise updater.UpdateError("The configured engine could not be resolved")
            return copy.deepcopy(self.dns[args[1]])
        if args[:2] == ("image", "inspect") and args[2] not in self.images:
            self.calls.append(("json", args))
            raise updater.UpdateError("No such image")
        return super().json(*args, **kwargs)

    def api(self, container_id, component, path, body=None, **kwargs):
        self.calls.append(("api", (container_id, component, path, body)))
        if not self.containers[container_id]["State"]["Running"]:
            raise updater.UpdateError("The container is stopped")
        if (container_id, path) in self.responses:
            return copy.deepcopy(self.responses[container_id, path])
        labels = self.containers[container_id]["Config"].get("Labels") or {}
        role = self.roles.get(container_id) or labels.get(
            "goodvision.update.manual_role"
        )
        if role != component:
            raise updater.ApiError(404)
        if component == "harness":
            if path == "/v1/ui/info":
                return {
                    "configured": True,
                    "analysis_modes": ["event", "face", "multi"],
                    "window": {"size_frames": 16, "stride_frames": 8},
                    "face": {"engine": "local"},
                    "deployment": {"autostart": False},
                }
            if path == "/v1/status":
                return {"analysis_state": "idle", "pipeline_state": "idle"}
            if path == "/v1/deployment/settings":
                return {"autostart": False}
        if component == "contour" and path == "/api/status":
            return {
                "service": "GoodVision Contour",
                "model_id": "goodvision-contour-1",
                "mode": "connected",
                "accepting_sessions": True,
                "active_sessions": 0,
                "requests": {"in_flight": 0},
            }
        if component == "engine" and path == "/health/perception/sam_live":
            return {"backend": "sam3.1_video", "enabled": True, "gpu_ready": True}
        raise updater.ApiError(404)


def assert_read_only(docker):
    for kind, args in docker.calls:
        assert kind != "create"
        if kind == "run":
            assert args[0] not in {
                "stop",
                "start",
                "rename",
                "rm",
                "load",
                "build",
                "run",
            }
            assert args[:2] not in {("image", "load"), ("image", "tag")}
        if kind == "api":
            assert args[3] is None


@pytest.mark.parametrize(
    "name", ["", " ", "container*", "camera;stop", "camera\nother"]
)
def test_invalid_manual_name_is_refused_without_touching_docker(tmp_path, name):
    docker = ManualDocker(tmp_path)
    with pytest.raises(updater.UpdateError):
        customer.find_named_installations("harness", name, docker=docker)
    assert not docker.calls


def test_exact_container_name_takes_priority_over_an_image_reference(tmp_path):
    docker = ManualDocker(tmp_path)
    target = docker.container_for("harness")
    target["Name"] = "/customer-camera"
    other = docker.add_container(
        "other-camera", "harness", reference="other-main:installed"
    )
    docker.images["customer-camera"] = other["Image"]
    choices = customer.find_named_installations(
        "harness", "/customer-camera", docker=docker
    )
    assert len(choices) == 1
    assert choices[0]["selection"]["ids"] == {"harness": target["Id"]}
    assert choices[0]["selection"]["names"] == {"harness": "customer-camera"}
    assert not any(
        kind == "json" and args[:2] == ("image", "inspect")
        for kind, args in docker.calls
    )
    assert_read_only(docker)


def test_image_lookup_returns_each_verified_installation_and_excludes_publisher(
    tmp_path,
):
    docker = ManualDocker(tmp_path)
    first = docker.container_for("harness")
    second = docker.add_container("second-camera", "harness")
    publisher = docker.add_container("same-image-publisher", "publisher")
    choices = customer.find_named_installations(
        "harness", "customer-main:installed", docker=docker
    )
    assert len(choices) == 2
    assert {c["selection"]["ids"]["harness"] for c in choices} == {
        first["Id"],
        second["Id"],
    }
    assert publisher["Id"] not in {c["selection"]["ids"]["harness"] for c in choices}
    assert_read_only(docker)


def test_exact_publisher_name_does_not_fall_back_to_a_valid_image(tmp_path):
    docker = ManualDocker(tmp_path)
    publisher = docker.add_container("publisher", "publisher")
    docker.images["publisher"] = docker.container_for("harness")["Image"]
    assert (
        customer.find_named_installations("harness", "publisher", docker=docker) == []
    )
    assert any(
        kind == "api" and args[0] == publisher["Id"] for kind, args in docker.calls
    )
    assert_read_only(docker)


@pytest.mark.parametrize(
    "path,response",
    [
        ("/v1/ui/info", {"configured": False, "error": "configuration unavailable"}),
        ("/v1/ui/info", {"configured": True, "analysis_modes": ["event"]}),
        (
            "/v1/ui/info",
            {
                "configured": True,
                "analysis_modes": ["event", "face", "multi"],
                "window": {"size_frames": True, "stride_frames": 8},
                "face": {"engine": "local"},
            },
        ),
        (
            "/v1/ui/info",
            {
                "configured": True,
                "analysis_modes": ["event", "face", "multi"],
                "window": {"size_frames": 16, "stride_frames": 0},
                "face": {"engine": "local"},
            },
        ),
        ("/v1/status", {"analysis_state": None}),
        ("/v1/deployment/settings", {"autostart": 1}),
    ],
)
def test_manual_harness_requires_supported_live_api_shapes(tmp_path, path, response):
    docker = ManualDocker(tmp_path)
    target = docker.container_for("harness")
    docker.responses[target["Id"], path] = response
    assert (
        customer.find_named_installations("harness", target["Name"], docker=docker)
        == []
    )
    assert_read_only(docker)


def test_labeled_container_lookup_keeps_every_lane_in_the_project(tmp_path):
    docker = ManualDocker(tmp_path, labeled=True, services=("harness", "harness-02"))
    target = docker.container_for("harness")
    choices = customer.find_named_installations(
        "harness", target["Name"], docker=docker
    )
    assert len(choices) == 1
    assert choices[0]["selection"] == {
        "component": "harness",
        "project": docker.project,
        "ids": {
            c["Config"]["Labels"][updater.LABEL + "service"]: c["Id"]
            for c in docker.containers.values()
        },
    }
    assert_read_only(docker)


def test_manual_contour_pairs_only_the_verified_dns_peer_on_a_shared_network(tmp_path):
    docker = ManualDocker(tmp_path, "contour")
    gateway, engine = docker.container_for("contour"), docker.container_for("engine")
    docker.add_container(
        "unrelated-engine", "engine", reference="customer-engine:installed"
    )
    choices = customer.find_named_installations(
        "contour", gateway["Name"], docker=docker
    )
    assert len(choices) == 1
    assert choices[0]["selection"]["ids"] == {
        "contour": gateway["Id"],
        "engine": engine["Id"],
    }
    assert_read_only(docker)


def test_unlabeled_contour_engine_name_resolves_its_proven_gateway_only(tmp_path):
    docker = ManualDocker(tmp_path, "contour")
    gateway, engine = docker.container_for("contour"), docker.container_for("engine")
    other_engine = docker.add_container(
        "other-engine", "engine", reference="other-engine:installed"
    )
    other_gateway = docker.add_container(
        "other-contour", "contour", reference="other-contour:installed"
    )
    docker.dns[other_gateway["Id"]] = [
        other_engine["NetworkSettings"]["Networks"]["customer-network"]["IPAddress"]
    ]
    choices = customer.find_named_installations(
        "contour", engine["Name"], docker=docker
    )
    assert len(choices) == 1
    assert choices[0]["selection"]["ids"] == {
        "contour": gateway["Id"],
        "engine": engine["Id"],
    }
    assert choices[0]["label"] == gateway["Name"].lstrip("/")
    assert_read_only(docker)


def test_unlabeled_contour_engine_image_excludes_unpaired_and_wrong_api_containers(
    tmp_path,
):
    docker = ManualDocker(tmp_path, "contour")
    gateway, engine = docker.container_for("contour"), docker.container_for("engine")
    docker.add_container(
        "unpaired-engine", "engine", reference="customer-engine:installed"
    )
    docker.add_container(
        "engine-image-publisher", "publisher", reference="customer-engine:installed"
    )
    fake_gateway = docker.add_container(
        "unverified-contour", "contour", reference="customer-contour:installed"
    )
    docker.dns[fake_gateway["Id"]] = docker.dns[gateway["Id"]]
    docker.responses[fake_gateway["Id"], "/api/status"] = {
        "service": "Other Service",
        "model_id": "goodvision-contour-1",
        "mode": "connected",
        "accepting_sessions": True,
        "requests": {"in_flight": 0},
    }
    choices = customer.find_named_installations(
        "contour", "customer-engine:installed", docker=docker
    )
    assert len(choices) == 1
    assert choices[0]["selection"]["ids"] == {
        "contour": gateway["Id"],
        "engine": engine["Id"],
    }
    assert_read_only(docker)


@pytest.mark.parametrize(
    "mismatch", ["network", "dns", "health", "ambiguous", "remote"]
)
def test_manual_contour_refuses_an_unproven_engine_pair(tmp_path, mismatch):
    docker = ManualDocker(tmp_path, "contour")
    gateway, engine = docker.container_for("contour"), docker.container_for("engine")
    if mismatch == "network":
        engine["NetworkSettings"]["Networks"] = {
            "other-network": {"IPAddress": "172.20.0.3"}
        }
    elif mismatch == "dns":
        docker.dns[gateway["Id"]] = ["172.20.0.99"]
    elif mismatch == "health":
        docker.responses[engine["Id"], "/health/perception/sam_live"] = {
            "backend": "other",
            "enabled": True,
            "gpu_ready": True,
        }
    elif mismatch == "ambiguous":
        other = docker.add_container(
            "other-engine", "engine", reference="customer-engine:installed"
        )
        docker.dns[gateway["Id"]].append(
            other["NetworkSettings"]["Networks"]["customer-network"]["IPAddress"]
        )
    else:
        del docker.dns[gateway["Id"]]
    assert (
        customer.find_named_installations("contour", gateway["Name"], docker=docker)
        == []
    )
    assert_read_only(docker)


@pytest.mark.parametrize("labeled", [False, True])
def test_selected_container_ids_are_rechecked_before_adoption(tmp_path, labeled):
    docker = ManualDocker(tmp_path, labeled=labeled)
    target = docker.container_for("harness")
    selection = customer.find_named_installations(
        "harness", target["Name"], docker=docker
    )[0]["selection"]
    original_id = target["Id"]
    changed_id = hashlib.sha256(b"recreated-container").hexdigest()
    target["Id"] = changed_id
    docker.containers[changed_id] = docker.containers.pop(original_id)
    docker.roles[changed_id] = docker.roles.pop(original_id)
    with pytest.raises(updater.UpdateError, match="changed"):
        discovery.named_deployment(
            docker, "harness", selection, state_directory=tmp_path
        )
    assert_read_only(docker)


def test_manual_installation_identity_is_stable_across_recreated_container_ids(
    tmp_path,
):
    docker = ManualDocker(tmp_path)
    target = docker.container_for("harness")
    first_selection = customer.find_named_installations(
        "harness", target["Name"], docker=docker
    )[0]["selection"]
    first = discovery.named_deployment(
        docker, "harness", first_selection, state_directory=tmp_path
    )
    original_id = target["Id"]
    changed_id = hashlib.sha256(b"updated-container").hexdigest()
    target["Id"] = changed_id
    docker.containers[changed_id] = docker.containers.pop(original_id)
    docker.roles[changed_id] = docker.roles.pop(original_id)
    second_selection = customer.find_named_installations(
        "harness", target["Name"], docker=docker
    )[0]["selection"]
    second = discovery.named_deployment(
        docker, "harness", second_selection, state_directory=tmp_path
    )
    assert first.project == second.project
    assert first.state_dir == second.state_dir
    assert first.manual_names == second.manual_names
    assert first_selection["ids"] != second_selection["ids"]
    assert_read_only(docker)


def test_install_rejects_stale_manual_choice_before_loading_or_stopping_anything(
    tmp_path,
):
    directory = tmp_path / "customer update"
    make_package(directory)
    docker = ManualDocker(tmp_path)
    target = docker.container_for("harness")
    selection = customer.find_named_installations(
        "harness", target["Name"], docker=docker
    )[0]["selection"]
    selection["ids"]["harness"] = hashlib.sha256(b"obsolete-container").hexdigest()
    with pytest.raises(customer.CustomerError, match="changed"):
        customer.install(directory, selection=selection, docker=docker)
    assert_read_only(docker)


def test_install_refuses_manual_choice_for_a_different_package_component(tmp_path):
    directory = tmp_path / "customer update"
    make_package(directory)
    docker = ManualDocker(tmp_path, "contour")
    gateway = docker.container_for("contour")
    selection = customer.find_named_installations(
        "contour", gateway["Name"], docker=docker
    )[0]["selection"]
    docker.calls.clear()
    with pytest.raises(customer.CustomerError, match="matching"):
        customer.install(directory, selection=selection, docker=docker)
    assert not docker.calls


def test_runtime_update_adopts_unlabeled_container_and_keeps_repeat_lookup_stable(
    tmp_path,
):
    docker = ManualDocker(tmp_path)
    original = docker.container_for("harness")
    original["Name"] = "/customer-camera"
    # Native Docker reports absent labels as null on older docker-run installs.
    original["Config"]["Labels"] = None
    first_choice = customer.find_named_installations(
        "harness", "customer-camera", docker=docker
    )[0]
    first = discovery.named_deployment(
        docker, "harness", first_choice["selection"], state_directory=tmp_path
    )
    first.state_dir.mkdir(parents=True)
    runtime.runtime_update(docker, first, docker.candidates, timeout=1)
    current = updater.current_containers(docker, first)["harness"]
    assert current["Name"] == "/customer-camera"
    assert current["Image"] == docker.candidates["harness"]
    assert (
        current["Config"]["Labels"]["goodvision.update.manual_project"] == first.project
    )
    assert current["Config"]["Labels"]["goodvision.update.manual_role"] == "harness"
    assert (
        current["HostConfig"]["DeviceRequests"]
        == original["HostConfig"]["DeviceRequests"]
    )
    assert docker.images["customer-main:installed"] == current["Image"]
    assert original["Id"] not in docker.containers
    second_choice = customer.find_named_installations(
        "harness", "customer-camera", docker=docker
    )[0]
    second = discovery.named_deployment(
        docker, "harness", second_choice["selection"], state_directory=tmp_path
    )
    assert second.project == first.project
    assert second.state_dir == first.state_dir
    assert second_choice["selection"]["ids"] == {"harness": current["Id"]}


def test_autostart_restore_does_not_write_to_an_unrelated_recreated_container(tmp_path):
    docker = ManualDocker(tmp_path)
    target = docker.container_for("harness")
    original_id = target["Id"]
    docker.responses[original_id, "/v1/deployment/settings"] = {"autostart": True}
    docker.responses[original_id, "/v1/analysis/stop"] = {}
    choice = customer.find_named_installations(
        "harness", target["Name"], docker=docker
    )[0]
    deployment = discovery.named_deployment(
        docker, "harness", choice["selection"], state_directory=tmp_path
    )
    unrelated_id = hashlib.sha256(b"unrelated-recreated-container").hexdigest()
    with pytest.raises(customer.CustomerError):
        with customer.stopped_harness(docker, deployment, lambda _message: None):
            replacement = copy.deepcopy(target)
            replacement["Id"] = unrelated_id
            del docker.containers[original_id]
            docker.containers[unrelated_id] = replacement
            docker.roles[unrelated_id] = "harness"
    assert not any(
        kind == "api" and args[0] == unrelated_id and args[3] is not None
        for kind, args in docker.calls
    )
