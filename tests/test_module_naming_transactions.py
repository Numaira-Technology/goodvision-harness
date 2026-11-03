import copy
import json
import subprocess
from pathlib import Path

import pytest

from tools import deploy_modules as deploy
from tools import docker_modules as spec


class NamingDocker:
    """Model Docker's recorded recipe, image identity, and existing resources."""

    def __init__(self, state, document):
        self.state = state
        self.document = copy.deepcopy(document)
        self.calls = []
        self.starts = []
        self.fail_next_up = False
        self.image_ids = {
            service["image"]: "sha256:" + str(i + 1) * 64
            for i, service in enumerate(document["services"].values())
        }
        self.running_ids = dict(self.image_ids)

    def __call__(self, command, **options):
        self.calls.append(command)
        value = ""
        if command[1:3] == ["image", "inspect"]:
            if command[-1] not in self.image_ids:
                raise subprocess.CalledProcessError(1, command)
            value = json.dumps([{"Id": self.image_ids[command[-1]]}])
        elif command[1] == "inspect":
            role, service = next((r, s) for r, s in self.document["services"].items()
                                 if s["container_name"] == command[-1])
            value = json.dumps([{
                "Image": self.running_ids[service["image"]],
                "Config": {"Labels": {
                    "com.docker.compose.project": self.document["name"],
                    "com.docker.compose.service": role,
                    "com.docker.compose.project.working_dir": str(self.state),
                    "io.goodvision.namespace": "office",
                }},
                "Mounts": [
                    {"Type": "volume", "Name": "customer_data", "Destination": "/data"},
                    {"Type": "volume", "Name": "customer_models",
                     "Destination": "/models"},
                ],
                "NetworkSettings": {"Networks": {"old_engine": {}}},
            }])
        elif command[1:3] == ["network", "inspect"]:
            value = json.dumps([{"Labels": {
                "com.docker.compose.project": self.document["name"],
                "com.docker.compose.network": "engine",
            }}])
        elif command[1] == "ps":
            if any(command[-1] == f"name=^/{s['container_name']}$"
                   for s in self.document["services"].values()):
                value = "existing-container-id\n"
        elif command[1] == "compose" and "up" in command:
            path = Path(command[command.index("-f") + 1])
            if self.fail_next_up:
                self.fail_next_up = False
                raise subprocess.CalledProcessError(1, command)
            self.document = json.loads(path.read_text())
            self.starts.append((path, copy.deepcopy(self.document)))
        return subprocess.CompletedProcess(command, 0, stdout=value, stderr="")


def setup_install(tmp_path, monkeypatch, module="main"):
    state = tmp_path.resolve() / "state"
    settings = spec.defaults("office", "v1")
    settings["module_versions"]["bbox"] = "independent"
    settings["module_images"] = {"bbox": {"bbox": "goodvision-bbox:independent-cpu"}}
    deploy.write_json(state / "settings.json", settings)
    document = spec.compose(module, settings, tmp_path, state)
    deploy.write_json(state / f"compose.{module}.json", document)
    docker = NamingDocker(state, document)
    monkeypatch.setattr(deploy, "location", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(deploy, "run", docker)
    monkeypatch.setattr(deploy, "ensure_network", lambda _: None)
    return state, settings, document, docker


@pytest.mark.parametrize("failure", [None, "health", "settings"])
def test_rename_mode_survives_configure_up_and_failed_migration(
    tmp_path, monkeypatch, failure,
):
    state, settings, before, docker = setup_install(tmp_path, monkeypatch)
    rollback = {"module": "main", "version": "v0", "compose": copy.deepcopy(before)}
    deploy.write_json(state / "rollback.main.json", rollback)
    # A newer delivery installer must not relabel any installed image version.
    monkeypatch.setattr(deploy, "read_package", lambda _: {
        "module": "main", "version": "v99", "files": {},
    })
    if failure == "health":
        docker.fail_next_up = True
    elif failure == "settings":
        original_write = deploy.write_json

        def fail_commit(path, value):
            if path.name == "settings.json" and value.get("namespaced_names"):
                raise OSError("injected settings write failure")
            original_write(path, value)

        monkeypatch.setattr(deploy, "write_json", fail_commit)
    args = ["--modules", "main", "--namespace", "office", "--state-dir", str(state)]
    assert deploy.main(["rename", "--namespaced-names", *args]) == int(bool(failure))
    saved = json.loads((state / "settings.json").read_text())
    if failure:
        assert saved == settings
        assert json.loads((state / "compose.main.json").read_text()) == before
        assert json.loads((state / "rollback.main.json").read_text()) == rollback
        assert docker.document == before
    else:
        assert saved == {**settings, "namespaced_names": True}
        migrated = json.loads((state / "compose.main.json").read_text())
        assert migrated["name"] == "goodvision-main-office"
        assert migrated["services"]["main"]["container_name"] == (
            "goodvision-main-office"
        )
        assert migrated["volumes"]["data"] == {
            "external": True, "name": "customer_data",
        }
        # Subsequent source CLI invocations omit the naming flag entirely.
        monkeypatch.setattr(deploy, "read_package", lambda _: None)
        for action in ("configure", "up"):
            assert deploy.main([action, *args]) == 0
            assert json.loads((state / "compose.main.json").read_text()) == migrated
            assert json.loads((state / "settings.json").read_text()) == saved
        starts = len(docker.starts)
        assert deploy.main(["rename", *args]) == 0
        assert len(docker.starts) == starts
    recorded_path, recorded_recipe = docker.starts[-1]
    assert recorded_path == state / "compose.main.json"
    assert json.loads(recorded_path.read_text()) == recorded_recipe
    assert not (state / "compose.main.renamed.json").exists()


@pytest.mark.parametrize("module", ["main", "contour"])
@pytest.mark.parametrize("drift", ["mismatch", "missing"])
def test_rename_rejects_image_drift_before_any_destructive_action(
    tmp_path, monkeypatch, module, drift,
):
    state, settings, before, docker = setup_install(tmp_path, monkeypatch, module)
    role = "engine" if module == "contour" else "main"
    image = before["services"][role]["image"]
    if drift == "missing":
        del docker.image_ids[image]
    else:
        docker.image_ids[image] = "sha256:" + "f" * 64
    contents = {p.name: p.read_bytes() for p in state.glob("*.json")}
    with pytest.raises(ValueError, match=r"missing|differs from the running"):
        deploy.rename_project(state, module, {**settings, "namespaced_names": True},
                              30, docker)
    assert {p.name: p.read_bytes() for p in state.glob("*.json")} == contents
    assert docker.document == before
    assert not any(call[1] == "compose" for call in docker.calls)
    assert not docker.starts


@pytest.mark.parametrize("failure", ["second_module", "settings_commit", "bad_record"])
def test_multi_module_rename_rolls_back_partial_success_and_can_retry(
    tmp_path, monkeypatch, failure,
):
    state, settings, _, main_docker = setup_install(tmp_path, monkeypatch)
    original_bbox = spec.compose("bbox", settings, tmp_path, state)
    deploy.write_json(state / "compose.bbox.json", original_bbox)
    bbox_docker = NamingDocker(state, original_bbox)
    engines = {"main": main_docker, "bbox": bbox_docker}
    before = {p.name: p.read_bytes() for p in state.glob("*.json")}
    # Changing shared mode cannot strand an omitted installed module.
    assert deploy.main(["rename", "--namespaced-names", "--modules", "main",
                        "--namespace", "office", "--state-dir", str(state)]) == 1
    assert not main_docker.calls
    for filename, data in before.items():
        assert (state / filename).read_bytes() == data

    def run(command, **options):
        module = "bbox" if any("bbox" in part for part in command[1:]) else "main"
        return engines[module](command, **options)

    monkeypatch.setattr(deploy, "run", run)
    original_write = deploy.write_json
    if failure == "second_module":
        bbox_docker.fail_next_up = True
    elif failure == "bad_record":
        deploy.write_json(state / "rollback.bbox.json", {"version": "broken"})
    else:
        def fail_commit(path, value):
            if path.name == "settings.json" and value.get("namespaced_names"):
                raise OSError("injected group commit failure")
            original_write(path, value)
        monkeypatch.setattr(deploy, "write_json", fail_commit)
    args = ["--modules", "main", "bbox", "--namespace", "office",
            "--state-dir", str(state)]
    assert deploy.main(["rename", "--namespaced-names", *args]) == 1
    for filename, data in before.items():
        assert (state / filename).read_bytes() == data
    for module, engine in engines.items():
        recipe = json.loads((state / f"compose.{module}.json").read_text())
        assert engine.document == recipe
        if engine.starts:
            assert engine.starts[-1] == (state / f"compose.{module}.json", recipe)
    if failure == "bad_record":
        record = state / "rollback.bbox.json"
        assert json.loads(record.read_text()) == {"version": "broken"}
        assert not bbox_docker.starts
        record.unlink()  # Owner repairs the bad record before retrying.
    assert not list(state.glob("rollback.*.json"))
    assert deploy.main(["configure", *args]) == 0

    monkeypatch.setattr(deploy, "write_json", original_write)
    assert deploy.main(["rename", "--namespaced-names", *args]) == 0
    assert json.loads((state / "settings.json").read_text()) == {
        **settings, "namespaced_names": True,
    }
    starts = {m: len(e.starts) for m, e in engines.items()}
    assert deploy.main(["rename", *args]) == 0
    assert starts == {m: len(e.starts) for m, e in engines.items()}
    for action in ("configure", "up"):
        assert deploy.main([action, *args]) == 0
        for module in engines:
            recipe = json.loads((state / f"compose.{module}.json").read_text())
            assert recipe["name"] == f"goodvision-{module}-office"
            assert recipe["volumes"]["data"]["name"] == "customer_data"
    assert json.loads((state / "settings.json").read_text())["module_versions"] == (
        settings["module_versions"]
    )
