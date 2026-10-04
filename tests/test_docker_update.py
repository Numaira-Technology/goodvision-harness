"""Exercise Compose update safety without a Docker daemon or application data."""

from __future__ import annotations

import copy
import json

import pytest

from tools import update_docker as updater


class FakeDocker:
    def __init__(self, directory, component="harness", services=None, build=True):
        self.directory = directory
        self.component = component
        self.project = "installed"
        self.files = (directory / "compose.yaml", directory / "gpu.yaml")
        for path in self.files:
            path.write_text("services: {}\n", encoding="utf-8")
        self.env_file = directory / ".env"
        self.env_file.write_text("PRIVATE_KEY=never-save-this-key\n", encoding="utf-8")
        self.services = services or (
            ("harness", "harness-2")
            if component == "harness"
            else ("contour", "engine")
        )
        self.config = {
            "services": {
                service: {"image": f"installed-{service}:latest"}
                | ({"build": {"context": "."}} if build else {})
                for service in (*self.services, "unrelated-worker")
            }
        }
        self.images = {
            config["image"]: f"sha256:original-{service}"
            for service, config in self.config["services"].items()
        }
        self.containers = {
            service: self.container(service) for service in self.config["services"]
        }
        self.calls = []
        self.generation = 0
        self.fail_up = False
        self.on_build = None
        self.harness_status = {
            "analysis_state": "idle",
            "pipeline_state": "idle",
            "processing_cleanup": "complete",
        }
        self.autostart = False
        self.contour_status = {
            "gpu_ready": True,
            "active_sessions": 0,
            "requests": {"in_flight": 0},
            "accepting_sessions": True,
        }

    def container(self, service, project=None):
        project = project or self.project
        return {
            "Id": f"old-{project}-{service}",
            "Name": f"/{project}-{service}-1",
            "Image": self.images[self.config["services"][service]["image"]],
            "State": {"Running": True, "Paused": False},
            "Config": {
                "Env": ["PRIVATE_KEY=never-save-this-key"],
                "Labels": {
                    updater.LABEL + "project": project,
                    updater.LABEL + "service": service,
                    updater.LABEL + "project.working_dir": str(self.directory),
                    updater.LABEL + "project.config_files": ",".join(
                        map(str, self.files)
                    ),
                    updater.LABEL + "project.environment_file": str(self.env_file),
                    updater.LABEL + "config-hash": f"hash-{service}",
                },
            },
            "Mounts": [{"Source": "keep-private-data", "Destination": "/data"}],
        }

    def deployment(self):
        return updater.Deployment(
            self.component,
            self.project,
            self.directory,
            self.files,
            (self.env_file,),
            {
                service: copy.deepcopy(self.containers[service])
                for service in self.services
            },
            copy.deepcopy(self.config),
        )

    def run(self, *args, cwd=None, timeout=60):
        self.calls.append(("run", args))
        if args[0] == "info":
            return "linux"
        if args[0] == "ps":
            project = args[-1].partition("project=")[2]
            return " ".join(
                value["Id"]
                for value in self.containers.values()
                if not project
                or value["Config"]["Labels"][updater.LABEL + "project"] == project
            )
        if args[:2] == ("image", "tag"):
            self.images[args[3]] = args[2]
            return ""
        if args[0] != "compose":
            raise AssertionError(args)
        if "version" in args:
            return "Docker Compose version fixture"
        if "config" in args and "--hash" in args:
            return f"{args[-1]} hash-{args[-1]}"
        if "build" in args:
            self.generation += 1
            for service in args[args.index("build") + 1 :]:
                reference = self.config["services"][service]["image"]
                self.images[reference] = f"sha256:build-{self.generation}-{service}"
            if self.on_build:
                self.on_build()
            return ""
        if "stop" in args:
            for service in args[args.index("stop") + 1 :]:
                self.containers[service]["State"]["Running"] = False
            return ""
        if "up" in args:
            if self.fail_up:
                self.fail_up = False
                raise updater.UpdateError("Replacement startup failed")
            overlay_path = self.deployment().overlay
            selected = json.loads(overlay_path.read_text(encoding="utf-8"))["services"]
            for service in args[args.index("--no-deps") + 1 :]:
                config = selected[service]
                container = self.containers[service]
                container["Id"] = f"replacement-{self.generation}-{service}"
                container["Image"] = config["image"]
                container["State"]["Running"] = True
                labels = container["Config"]["Labels"]
                labels[updater.LABEL + "project.config_files"] = ",".join(
                    map(str, (*self.files, overlay_path))
                )
            if self.component == "contour":
                self.contour_status["active_sessions"] = 0
            return ""
        raise AssertionError(args)

    def json(self, *args, **kwargs):
        self.calls.append(("json", args))
        if args[:2] == ("context", "inspect"):
            return [{"Endpoints": {"docker": {"Host": "unix:///var/run/docker.sock"}}}]
        if args[0] == "inspect":
            return copy.deepcopy(
                [value for value in self.containers.values() if value["Id"] in args[1:]]
            )
        if args[:2] == ("image", "inspect"):
            reference = args[2]
            if reference.startswith("sha256:"):
                return [{"Id": reference}]
            if reference not in self.images:
                raise updater.UpdateError("Candidate image is not loaded")
            return [{"Id": self.images[reference]}]
        if args[0] == "compose" and "config" in args:
            config = copy.deepcopy(self.config)
            overlay_path = self.deployment().overlay
            if str(overlay_path) in args:
                for service, value in json.loads(
                    overlay_path.read_text(encoding="utf-8")
                )["services"].items():
                    config["services"][service].update(value)
            return config
        raise AssertionError(args)

    def api(self, identity, component, path, body=None):
        self.calls.append(("api", (identity, component, path, body)))
        if component == "harness":
            if path == "/v1/status":
                return copy.deepcopy(self.harness_status)
            if path == "/v1/deployment/settings":
                return {"autostart": self.autostart}
            assert path == "/v1/ui/info"
            return {"version": "fixture"}
        if path == "/api/admission":
            self.contour_status.update(body)
        else:
            assert path == "/api/status"
        return copy.deepcopy(self.contour_status)

    @property
    def mutations(self):
        return [
            args
            for kind, args in self.calls
            if (
                (
                    kind == "run"
                    and (
                        args[:2] == ("image", "tag")
                        or (
                            args[0] == "compose"
                            and any(
                                action in args for action in ("build", "stop", "up")
                            )
                        )
                    )
                )
                or (kind == "api" and args[-1] is not None)
            )
        ]


def test_discovery_requires_unambiguous_project_without_mutation(tmp_path):
    docker = FakeDocker(tmp_path)
    docker.containers["second"] = docker.container("harness", "another")
    with pytest.raises(updater.UpdateError, match="--project"):
        updater.discover(docker, "harness")
    assert not docker.mutations
    deployment = updater.discover(docker, "harness", "installed")
    assert deployment.services == ["harness", "harness-2"]
    assert deployment.files == docker.files
    assert deployment.env_files == (docker.env_file,)


@pytest.mark.parametrize("case", ["replica", "files", "missing-engine"])
def test_invalid_installation_is_rejected_before_mutation(tmp_path, case):
    docker = FakeDocker(tmp_path, "contour")
    if case == "replica":
        duplicate = docker.container("contour")
        duplicate["Id"] = "duplicate"
        docker.containers["duplicate"] = duplicate
    elif case == "files":
        docker.containers["engine"]["Config"]["Labels"][
            updater.LABEL + "project.config_files"
        ] = str(tmp_path / "other.yaml")
    else:
        del docker.containers["engine"]
    with pytest.raises(updater.UpdateError):
        updater.discover(docker, "contour")
    assert not docker.mutations


@pytest.mark.parametrize(
    "case", ["analysis", "pipeline", "cleanup", "autostart", "stopped", "paused"]
)
def test_harness_must_be_stopped_and_autostart_disabled(tmp_path, case):
    docker = FakeDocker(tmp_path)
    if case == "analysis":
        docker.harness_status["analysis_state"] = "running"
    elif case == "pipeline":
        docker.harness_status["pipeline_state"] = "draining"
    elif case == "cleanup":
        docker.harness_status["processing_cleanup"] = "closing"
    elif case == "autostart":
        docker.autostart = True
    else:
        docker.containers["harness-2"]["State"][
            "Paused" if case == "paused" else "Running"
        ] = case == "paused"
    with pytest.raises(updater.UpdateError):
        updater.update(docker, docker.deployment())
    assert not docker.mutations
    assert not docker.deployment().state_dir.exists()


@pytest.mark.parametrize("field", ["active_sessions", "in_flight"])
def test_contour_busy_requests_refuse_update_without_pausing(tmp_path, field):
    docker = FakeDocker(tmp_path, "contour")
    if field == "active_sessions":
        docker.contour_status[field] = 1
    else:
        docker.contour_status["requests"][field] = 1
    with pytest.raises(updater.UpdateError, match="sessions"):
        updater.update(docker, docker.deployment())
    assert not docker.mutations


@pytest.mark.parametrize("component", ["harness", "contour"])
def test_check_only_does_not_create_state_or_mutate_docker(tmp_path, component):
    docker = FakeDocker(tmp_path, component)
    deployment = updater.discover(docker, component)
    updater.update(docker, deployment, check_only=True)
    assert not docker.mutations
    assert not deployment.state_dir.exists()


def test_all_selected_lanes_build_before_any_service_is_stopped(tmp_path):
    docker = FakeDocker(tmp_path)
    unrelated = copy.deepcopy(docker.containers["unrelated-worker"])
    deployment = docker.deployment()
    updater.update(docker, deployment)
    commands = [args for kind, args in docker.calls if kind == "run"]
    build = next(args for args in commands if "build" in args)
    stop = next(args for args in commands if "stop" in args)
    assert commands.index(build) < commands.index(stop)
    assert build[build.index("build") + 1 :] == tuple(deployment.services)
    assert stop[stop.index("stop") + 1 :] == tuple(deployment.services)
    assert docker.containers["unrelated-worker"] == unrelated
    for command in (args for args in commands if "up" in args):
        assert "--no-deps" in command
        assert "--no-build" in command
        assert "--pull" in command and "never" in command
    assert not (deployment.state_dir / "update.lock").exists()
    state = json.loads(deployment.record.read_text(encoding="utf-8"))
    assert state["status"] == "ready"
    saved = "\n".join(
        path.read_text(encoding="utf-8") for path in deployment.state_dir.iterdir()
    )
    assert "never-save-this-key" not in saved
    assert "PRIVATE_KEY" not in saved
    assert "keep-private-data" not in saved
    assert "environment" not in saved


def test_analysis_starting_during_build_prevents_replacement(tmp_path):
    docker = FakeDocker(tmp_path)
    docker.on_build = lambda: docker.harness_status.update(analysis_state="starting")
    deployment = docker.deployment()
    with pytest.raises(updater.UpdateError, match="Stop all analysis"):
        updater.update(docker, deployment)
    assert all("stop" not in args and "up" not in args for args in docker.mutations)
    assert not deployment.record.exists()
    assert not deployment.overlay.exists()


@pytest.mark.parametrize("accepting", [True, False])
def test_contour_pauses_after_build_and_replaces_both_services(tmp_path, accepting):
    docker = FakeDocker(tmp_path, "contour")
    docker.contour_status["accepting_sessions"] = accepting
    deployment = docker.deployment()
    updater.update(docker, deployment)
    build_index = next(
        i
        for i, call in enumerate(docker.calls)
        if call[0] == "run" and "build" in call[1]
    )
    pause_index = next(
        i
        for i, call in enumerate(docker.calls)
        if call[0] == "api" and call[1][-1] is not None
    )
    stop_index = next(
        i
        for i, call in enumerate(docker.calls)
        if call[0] == "run" and "stop" in call[1]
    )
    assert build_index < pause_index < stop_index
    stops = [
        args[args.index("stop") + 1 :]
        for kind, args in docker.calls
        if kind == "run" and "stop" in args
    ]
    assert stops == [("contour", "engine")]
    assert docker.contour_status["accepting_sessions"] is accepting
    admission = [
        args[-1]
        for kind, args in docker.calls
        if kind == "api" and args[2] == "/api/admission"
    ]
    assert admission[-1] == {"accepting_sessions": accepting}
    assert all(value == {"accepting_sessions": False} for value in admission[:-1])
    assert len(admission) >= 3  # Old gateway, new gateway, restore original state.


@pytest.mark.parametrize("failure", ["up", "readiness"])
def test_failed_replacement_restores_image_ids_after_build_moves_tags(
    tmp_path, monkeypatch, failure
):
    docker = FakeDocker(tmp_path, "contour")
    deployment = docker.deployment()
    before = {
        service: container["Image"]
        for service, container in deployment.containers.items()
    }
    if failure == "up":
        docker.fail_up = True
    else:
        real_wait = updater.wait_ready
        calls = []

        def fail_first(docker, deployment, timeout):
            calls.append(True)
            if len(calls) == 1:
                raise updater.UpdateError("GPU warmup failed")
            real_wait(docker, deployment, timeout)

        monkeypatch.setattr(updater, "wait_ready", fail_first)
    with pytest.raises(updater.UpdateError, match="previous images were restored"):
        updater.update(docker, deployment)
    assert {
        service: docker.containers[service]["Image"] for service in deployment.services
    } == before
    overlay = json.loads(deployment.overlay.read_text(encoding="utf-8"))
    assert {
        service: value["image"] for service, value in overlay["services"].items()
    } == before
    assert all(
        docker.images[docker.config["services"][service]["image"]] != image
        for service, image in before.items()
    )
    assert docker.contour_status["accepting_sessions"] is True
    assert not (deployment.state_dir / "update.lock").exists()
    assert json.loads(deployment.record.read_text(encoding="utf-8"))["before"] == before


def test_repeated_updates_discover_original_files_and_rollback_latest(tmp_path):
    docker = FakeDocker(tmp_path)
    first = updater.discover(docker, "harness")
    updater.update(docker, first)
    first_images = {
        service: docker.containers[service]["Image"] for service in first.services
    }
    second = updater.discover(docker, "harness")
    assert second.files == docker.files
    updater.update(docker, second)
    second_images = {
        service: docker.containers[service]["Image"] for service in second.services
    }
    assert first_images != second_images
    state = json.loads(second.record.read_text(encoding="utf-8"))
    assert state["before"] == first_images
    assert state["after"] == second_images
    updater.update(docker, updater.discover(docker, "harness"), rollback=True)
    assert {
        service: docker.containers[service]["Image"] for service in second.services
    } == first_images


def test_image_only_installation_requires_complete_local_candidates(tmp_path):
    docker = FakeDocker(tmp_path, build=False)
    deployment = docker.deployment()
    with pytest.raises(updater.UpdateError, match="no build definition"):
        updater.update(docker, deployment)
    with pytest.raises(updater.UpdateError, match="every selected service"):
        updater.update(docker, deployment, images={"harness": "loaded-candidate"})
    assert not docker.mutations
    images = {}
    for service in deployment.services:
        reference = f"candidate-{service}:v2"
        docker.images[reference] = f"sha256:candidate-{service}"
        images[service] = reference
    updater.update(docker, deployment, images=images)
    assert not any("build" in args for kind, args in docker.calls if kind == "run")
    assert all(
        docker.containers[service]["Image"] == f"sha256:candidate-{service}"
        for service in deployment.services
    )


def test_modified_environment_prevents_saved_rollback(tmp_path):
    docker = FakeDocker(tmp_path)
    deployment = docker.deployment()
    updater.update(docker, deployment)
    docker.calls.clear()
    docker.env_file.write_text("PRIVATE_KEY=changed\n", encoding="utf-8")
    with pytest.raises(updater.UpdateError, match="files changed"):
        updater.update(docker, deployment, rollback=True)
    assert not docker.mutations


def test_check_only_rollback_does_not_write_or_stop(tmp_path):
    docker = FakeDocker(tmp_path)
    deployment = docker.deployment()
    updater.update(docker, deployment)
    state_before = {path: path.read_bytes() for path in deployment.state_dir.iterdir()}
    docker.calls.clear()
    updater.update(docker, deployment, rollback=True, check_only=True)
    assert not docker.mutations
    assert {
        path: path.read_bytes() for path in deployment.state_dir.iterdir()
    } == state_before


@pytest.mark.parametrize("change", ["files", "environment", "containers"])
def test_drift_during_build_prevents_stopping_installed_services(tmp_path, change):
    docker = FakeDocker(tmp_path)
    deployment = docker.deployment()

    def drift():
        if change == "files":
            docker.files[1].write_text("changed configuration", encoding="utf-8")
        elif change == "environment":
            docker.config["services"]["harness"]["environment"] = {"PORT": "9999"}
        else:
            docker.containers["harness"]["Id"] = "installed-by-another-updater"

    docker.on_build = drift
    with pytest.raises(updater.UpdateError, match="changed while preparing"):
        updater.update(docker, deployment)
    assert not any("stop" in args or "up" in args for args in docker.mutations)
    assert not deployment.record.exists()
    assert not deployment.overlay.exists()


def test_resolved_environment_drift_prevents_saved_rollback(tmp_path):
    docker = FakeDocker(tmp_path)
    deployment = docker.deployment()
    updater.update(docker, deployment)
    docker.calls.clear()
    docker.config["services"]["harness"]["environment"] = {"PORT": "9999"}
    with pytest.raises(updater.UpdateError, match="files changed"):
        updater.update(docker, deployment, rollback=True)
    assert not docker.mutations


def test_external_image_upgrade_prevents_saved_rollback(tmp_path):
    docker = FakeDocker(tmp_path)
    updater.update(docker, docker.deployment())
    docker.calls.clear()
    docker.containers["harness"]["Image"] = "sha256:newer-external-update"
    with pytest.raises(updater.UpdateError, match="outside this updater"):
        updater.update(docker, docker.deployment(), rollback=True)
    assert not docker.mutations


def test_active_overlay_keeps_other_services_pinned(tmp_path):
    docker = FakeDocker(tmp_path)
    deployment = docker.deployment()
    deployment.state_dir.mkdir(parents=True)
    updater.write_json(
        deployment.overlay,
        {"services": {"unrelated-worker": {"image": "sha256:pinned-unrelated"}}},
    )
    unrelated = copy.deepcopy(docker.containers["unrelated-worker"])
    updater.update(docker, deployment)
    overlay = json.loads(deployment.overlay.read_text(encoding="utf-8"))
    assert overlay["services"]["unrelated-worker"] == {
        "image": "sha256:pinned-unrelated"
    }
    assert docker.containers["unrelated-worker"] == unrelated


def test_existing_update_lock_prevents_mutation_and_is_retained(tmp_path):
    docker = FakeDocker(tmp_path)
    deployment = docker.deployment()
    deployment.state_dir.mkdir(parents=True)
    lock = deployment.state_dir / "update.lock"
    lock.write_text("other updater", encoding="utf-8")
    with pytest.raises(updater.UpdateError, match="update lock exists"):
        updater.update(docker, deployment)
    assert not docker.mutations
    assert lock.read_text(encoding="utf-8") == "other updater"


def test_contour_rollback_recovers_stopped_engine_with_unknown_sessions(tmp_path):
    docker = FakeDocker(tmp_path, "contour")
    updater.update(docker, docker.deployment())
    before = json.loads(docker.deployment().record.read_text(encoding="utf-8"))[
        "before"
    ]
    docker.containers["engine"]["State"]["Running"] = False
    docker.contour_status["active_sessions"] = None
    updater.update(docker, docker.deployment(), rollback=True)
    assert {
        service: docker.containers[service]["Image"] for service in docker.services
    } == before
    assert all(
        docker.containers[service]["State"]["Running"] for service in docker.services
    )
    assert docker.contour_status["accepting_sessions"] is True


def test_failed_recovery_retains_original_error_if_admission_restore_fails(
    tmp_path, monkeypatch, capsys
):
    docker = FakeDocker(tmp_path, "contour")
    original_api = docker.api

    def fail_restore(identity, component, path, body=None):
        if body == {"accepting_sessions": True}:
            raise updater.UpdateError("Restore admission failed")
        return original_api(identity, component, path, body)

    def fail_readiness(*args):
        raise updater.UpdateError("Engine failed to start")

    monkeypatch.setattr(docker, "api", fail_restore)
    monkeypatch.setattr(updater, "wait_ready", fail_readiness)
    with pytest.raises(updater.UpdateError, match="Recovery did not become ready"):
        updater.update(docker, docker.deployment())
    assert "Could not restore Contour admission" in capsys.readouterr().err
    assert docker.deployment().record.is_file()
    assert not (docker.deployment().state_dir / "update.lock").exists()
