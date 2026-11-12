"""Independent CLIP installation preserves the original deployment and identities."""

from __future__ import annotations

import copy
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from tools import neural_container as neural

IMAGE = {
    "name": "goodvision-neural:v2.0-test",
    "id": "sha256:" + "a" * 64,
    "config_id": "sha256:" + "c" * 64,
    "os": "linux",
    "architecture": "amd64",
}
BASE = {
    "name": "goodvision-system1:pr75-603f20d",
    "id": "sha256:" + "b" * 64,
    "config_id": "sha256:" + "d" * 64,
    "os": "linux",
    "architecture": "amd64",
}


def archive(path, image=IMAGE, *, extra=None):
    entries = [
        {
            "digest": image["id"],
            "annotations": {
                "io.containerd.image.name": "docker.io/library/" + image["name"],
                "org.opencontainers.image.ref.name": image["name"].split(":", 1)[1],
            },
        }
    ]
    if extra:
        entries.append(extra)
    with tarfile.open(path, "w") as output:
        for name, value in (
            ("index.json", {"schemaVersion": 2, "manifests": entries}),
            ("oci-layout", {"imageLayoutVersion": "1.0.0"}),
        ):
            data = json.dumps(value).encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))


@pytest.fixture
def kit(tmp_path):
    goodvision = tmp_path / "Original GoodVision"
    state = goodvision / "state"
    state.mkdir(parents=True)
    settings = {
        "namespace": "custom-site",
        "bind": "127.0.0.7",
        "gpu_id": "2",
        "ports": {"main": 28080, "system1": 28097, "bbox": 28096, "contour": 28093},
        "urls": {"neural": "http://custom-clip:8098"},
        "module_versions": {"main": "v2.1", "system1": "v2.0"},
    }
    neural.write_json(state / "settings.json", settings)
    neural.write_json(
        state / "compose.main.json",
        {
            "services": {
                "main": {
                    "image": "goodvision-main:existing",
                    "networks": ["shared"],
                    "environment": {
                        "GOODVISION_NEURAL_LOCAL_URL": "http://custom-clip:8098"
                    },
                }
            },
            "networks": {"shared": {"external": True, "name": "site-private-network"}},
        },
    )
    (state / "secrets").mkdir()
    (state / "secrets/contour-key").write_text("private-key-kept")
    snapshot = (
        state
        / "models/neural/huggingface/models--openai--clip-vit-large-patch14"
        / "snapshots"
        / neural.REVISION
    )
    snapshot.mkdir(parents=True)
    for name, size in neural.MODEL_FILES.items():
        with (snapshot / name).open("wb") as stream:
            stream.truncate(size)
    with (state / "models/neural/yolo26s.pt").open("wb") as stream:
        stream.truncate(20422725)
    directory = tmp_path / "New Neural V2.0"
    directory.mkdir()
    payload = directory / "neural.tar"
    archive(payload)
    info = {
        "module": neural.MODULE,
        "version": "v2.0",
        "image": IMAGE,
        "requires": [BASE],
        "image_archive": {
            "file": payload.name,
            "bytes": payload.stat().st_size,
            "sha256": neural.digest(payload),
        },
    }
    info["source_files"] = []
    root = Path(__file__).resolve().parents[1]
    for name in ("Dockerfile", "healthcheck.py"):
        target = directory / "source/workers/neural" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / "workers/neural" / name).read_bytes())
        info["source_files"].append(
            {
                "file": str(target.relative_to(directory)),
                "bytes": target.stat().st_size,
                "sha256": neural.digest(target),
            }
        )
    for name in ("source/README.md", "source/tools/neural_container.py"):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Additional packaged source\n")
        info["source_files"].append(
            {
                "file": name,
                "bytes": target.stat().st_size,
                "sha256": neural.digest(target),
            }
        )
    neural.write_json(directory / "RELEASE.json", info)
    return goodvision, state, directory, info


class FakeDocker:
    def __init__(self, state):
        self.state = state
        self.calls = []
        self.started = False
        self.foreign = False
        self.fail_up = False
        self.wrong_base = False
        self.built = False
        self.classic = False
        self.project_override = None
        self.alias_member = None

    def run(self, args, **kwargs):
        self.calls.append(args)
        if "up" in args:
            if self.fail_up:
                raise subprocess.CalledProcessError(1, args)
            self.started = True
        if args[0] == "build":
            self.built = True
        return subprocess.CompletedProcess(args, 0, "")

    def json(self, args):
        self.calls.append(args)
        if args[0] == "info":
            return {"OSType": "linux", "Architecture": "x86_64"}
        if args[:2] == ["network", "inspect"]:
            return [
                {"Containers": {"alias-container": {}} if self.alias_member else {}}
            ]
        if args[0] == "inspect":
            return [self.alias_member]
        if args[:2] == ["image", "inspect"]:
            baseline = args[2] == BASE["name"]
            identity = BASE if baseline else IMAGE
            identifier = identity["config_id"] if self.classic else identity["id"]
            if baseline and self.wrong_base:
                identifier = "wrong-base"
            if not baseline and self.built:
                identifier = "sha256:" + "e" * 64
            return [
                {
                    "Id": identifier,
                    "Os": "linux",
                    "Architecture": "amd64",
                    "RootFS": {
                        "Layers": ["base-layer"]
                        if baseline
                        else ["base-layer", "neural-layer"]
                    },
                    "Config": {
                        "Entrypoint": [
                            "python",
                            "-m",
                            "goodvision_harness.neural_service",
                        ],
                        "Cmd": [
                            "--host",
                            "0.0.0.0",
                            "--port",
                            "8098",
                            "--cache-dir",
                            "/models/huggingface",
                        ],
                        "Labels": {
                            "com.goodvision.module": neural.MODULE,
                            "org.opencontainers.image.version": "v2.0",
                        },
                        "Healthcheck": {
                            "Test": [
                                "CMD",
                                "python",
                                "/opt/goodvision-neural/healthcheck.py",
                            ]
                        },
                    },
                }
            ]
        raise AssertionError(args)

    def image(self, entry):
        return neural.Docker.image(self, entry)

    def container(self):
        if not self.started and not self.foreign:
            return None
        image_id = (
            "sha256:" + "e" * 64
            if self.built
            else IMAGE["config_id"]
            if self.classic
            else IMAGE["id"]
        )
        return {
            "Id": "only-neural-id",
            "Name": "/goodvision-neural",
            "Image": image_id,
            "State": {"Running": True, "Health": {"Status": "healthy"}},
            "Config": {
                "Labels": {
                    neural.LABEL: "another-installation"
                    if self.foreign
                    else str(self.state),
                    "com.docker.compose.project": neural.MODULE,
                    "com.docker.compose.service": "neural",
                }
            },
        }

    def project_members(self):
        if self.project_override is not None:
            return self.project_override
        container = self.container()
        return [container] if container else []


def test_custom_network_gpu_bind_and_model_reused(kit):
    goodvision, state, _, info = kit
    actual_state, document = neural.deployment(goodvision, info)
    service = document["services"]["neural"]
    assert actual_state == state
    assert document["networks"]["modules"]["name"] == "site-private-network"
    assert service["networks"]["modules"]["aliases"] == ["custom-clip"]
    assert service["deploy"]["resources"]["reservations"]["devices"][0][
        "device_ids"
    ] == ["2"]
    assert service["ports"][0]["host_ip"] == "127.0.0.7"
    assert service["volumes"][0]["source"] == str(state / "models/neural")
    assert service["volumes"][0]["read_only"] is True
    assert service["container_name"] == "goodvision-neural"
    assert service["restart"] == "unless-stopped"
    assert service["environment"]["HF_HUB_OFFLINE"] == "1"


@pytest.mark.parametrize("classic", [False, True])
def test_install_and_restart_touch_only_neural(kit, classic):
    goodvision, state, directory, _ = kit
    protected = {
        path: path.read_bytes()
        for path in (
            state / "settings.json",
            state / "compose.main.json",
            state / "secrets/contour-key",
        )
    }
    docker = FakeDocker(state)
    docker.classic = classic
    neural.install(goodvision, directory, docker)
    marker = neural.read(state / "neural-deployment.json")
    assert marker["status"] == "healthy"
    assert marker["gpu_inference"] == "service_warmup_and_health_passed"
    assert all(path.read_bytes() == data for path, data in protected.items())
    assert any(
        call[:2] == ["network", "inspect"] and call[-1] == "site-private-network"
        for call in docker.calls
    )
    up = next(call for call in docker.calls if "up" in call)
    assert up[-1] == "neural" and "--no-deps" in up and "--remove-orphans" not in up
    before = len([call for call in docker.calls if call[0] == "load"])
    neural.install(goodvision, directory, docker, start_only=True)
    assert len([call for call in docker.calls if call[0] == "load"]) == before
    assert not any(call[0] in ("stop", "rm", "restart") for call in docker.calls)


def test_wrong_baseline_prevents_loading_and_writing(kit):
    goodvision, state, directory, _ = kit
    docker = FakeDocker(state)
    docker.wrong_base = True
    with pytest.raises(ValueError, match="Required image identity differs"):
        neural.install(goodvision, directory, docker)
    assert not (state / "compose.neural.json").exists()
    assert not any(call[0] == "load" for call in docker.calls)


def test_foreign_container_is_never_adopted(kit):
    goodvision, state, directory, _ = kit
    docker = FakeDocker(state)
    docker.foreign = True
    with pytest.raises(ValueError, match="another installation"):
        neural.install(goodvision, directory, docker)
    assert not (state / "compose.neural.json").exists()
    assert not any(call[0] in ("load", "build") for call in docker.calls)


@pytest.mark.parametrize(
    "mutation", ["renamed", "foreign", "extra_service", "duplicate"]
)
def test_foreign_or_extra_compose_project_members_are_not_recreated(kit, mutation):
    goodvision, state, directory, _ = kit
    docker = FakeDocker(state)
    docker.started = True
    member = docker.container()
    docker.started = False
    if mutation == "renamed":
        member["Name"] = "/other-clip"
    elif mutation == "foreign":
        member["Config"]["Labels"][neural.LABEL] = "other-installation"
    elif mutation == "extra_service":
        member["Config"]["Labels"]["com.docker.compose.service"] = "other-service"
    docker.project_override = (
        [member, copy.deepcopy(member)] if mutation == "duplicate" else [member]
    )
    with pytest.raises(ValueError):
        neural.install(goodvision, directory, docker)
    assert not (state / "compose.neural.json").exists()
    assert not any(
        call[0] in ("load", "build") or "up" in call for call in docker.calls
    )


@pytest.mark.parametrize("owned", [False, True])
def test_duplicate_network_alias_requires_owned_container(kit, owned):
    goodvision, state, directory, _ = kit
    docker = FakeDocker(state)
    docker.started = True
    member = docker.container()
    docker.started = False
    member["NetworkSettings"] = {
        "Networks": {
            "site-private-network": {"Aliases": ["custom-clip"]},
        }
    }
    if not owned:
        member["Name"] = "/temporary-clip-check"
        member["Config"]["Labels"] = {}
    docker.alias_member = member
    if owned:
        neural.install(goodvision, directory, docker)
        assert neural.read(state / "neural-deployment.json")["status"] == "healthy"
    else:
        with pytest.raises(ValueError, match="alias is already used"):
            neural.install(goodvision, directory, docker)
        assert not (state / "compose.neural.json").exists()
        assert not any(
            call[0] in ("load", "build") or "up" in call for call in docker.calls
        )


def test_localhost_url_is_rejected_even_when_settings_agree(kit):
    goodvision, state, _, info = kit
    settings = neural.read(state / "settings.json")
    settings["urls"]["neural"] = "http://localhost:8098"
    neural.write_json(state / "settings.json", settings)
    document = neural.read(state / "compose.main.json")
    document["services"]["main"]["environment"]["GOODVISION_NEURAL_LOCAL_URL"] = (
        "http://localhost:8098"
    )
    neural.write_json(state / "compose.main.json", document)
    with pytest.raises(ValueError, match="service alias"):
        neural.deployment(goodvision, info)


def test_failed_startup_is_recorded_as_failed(kit):
    goodvision, state, directory, _ = kit
    docker = FakeDocker(state)
    docker.fail_up = True
    with pytest.raises(subprocess.CalledProcessError):
        neural.install(goodvision, directory, docker)
    assert neural.read(state / "neural-deployment.json")["status"] == "startup_failed"


def test_customized_neural_config_is_not_overwritten(kit):
    goodvision, state, directory, _ = kit
    neural.write_json(state / "compose.neural.json", {"keep": "custom"})
    with pytest.raises(ValueError, match="left unchanged"):
        neural.install(goodvision, directory, FakeDocker(state))
    assert neural.read(state / "compose.neural.json") == {"keep": "custom"}


@pytest.mark.parametrize(
    "mutation",
    ["main_url", "missing_model", "missing_person_model", "ambiguous_network"],
)
def test_inconsistent_original_state_fails_before_docker(kit, mutation):
    goodvision, state, _, info = kit
    if mutation == "main_url":
        settings = neural.read(state / "settings.json")
        settings["urls"]["neural"] = "http://different:8098"
        neural.write_json(state / "settings.json", settings)
    elif mutation == "missing_model":
        next((state / "models/neural").rglob("config.json")).unlink()
    elif mutation == "missing_person_model":
        (state / "models/neural/yolo26s.pt").unlink()
    else:
        document = neural.read(state / "compose.main.json")
        document["services"]["main"]["networks"].append("second")
        document["networks"]["second"] = {"external": True, "name": "second-network"}
        neural.write_json(state / "compose.main.json", document)
    with pytest.raises(ValueError):
        neural.deployment(goodvision, info)


def test_dry_run_writes_nothing_and_calls_no_docker(kit, capsys):
    goodvision, state, directory, _ = kit
    neural.main(
        [
            "install",
            "--dry-run",
            "--goodvision-dir",
            str(goodvision),
            "--release-dir",
            str(directory),
            "--docker",
            "must-not-execute",
        ]
    )
    assert json.loads(capsys.readouterr().out)["changes_existing_modules"] is False
    assert not (state / ".installer.lock").exists()
    assert not (state / "compose.neural.json").exists()


def test_archive_cannot_import_other_module_tags(tmp_path):
    path = tmp_path / "wrong.tar"
    archive(
        path,
        extra={
            "digest": BASE["id"],
            "annotations": {
                "io.containerd.image.name": BASE["name"],
            },
        },
    )
    with pytest.raises(ValueError, match="only the independent neural image"):
        neural.validate_archive(path, IMAGE)


def test_offline_build_records_actual_identity_and_never_loads(kit):
    goodvision, state, directory, info = kit
    docker = FakeDocker(state)
    docker.classic = True
    neural.install(goodvision, directory, docker, offline_build=True)
    saved = neural.read(state / "neural-deployment.json")
    assert saved["install_mode"] == "offline-build"
    assert saved["actual_image"]["id"] == "sha256:" + "e" * 64
    assert saved["image"]["id"] == info["image"]["id"]
    build = next(call for call in docker.calls if call[0] == "build")
    assert build[build.index("--network") + 1] == "none"
    assert "--pull=false" in build
    assert not any(call[0] == "load" for call in docker.calls)
    neural.install(goodvision, directory, docker, start_only=True)


def test_source_change_blocks_offline_build(kit):
    goodvision, state, directory, _ = kit
    (directory / "source/workers/neural/Dockerfile").write_text("FROM wrong\n")
    docker = FakeDocker(state)
    with pytest.raises(ValueError, match="source differs"):
        neural.install(goodvision, directory, docker, offline_build=True)
    assert not any(call[0] == "build" for call in docker.calls)


def test_unrelated_layers_fail_build_validation():
    found = FakeDocker(Path("/unused")).json(["image", "inspect", IMAGE["name"]])[0]
    base = copy.deepcopy(found)
    base["RootFS"]["Layers"] = ["different-baseline"]
    with pytest.raises(ValueError, match="preserve the verified base"):
        neural.verify_built_image(found, base)
