import copy
import json
import os
import subprocess
import sys
import zipfile
from urllib.parse import parse_qs, urlsplit

import pytest

from tools.harness_update.apply_update import replacement_config, upgrade
from tools.harness_update.package_update import BASE_ID, WINDOWS_BASE_IDS, package
from tools.harness_update.release_metadata import (
    read_metadata,
    update_metadata,
)


def container(index):
    return {
        "Id": f"old{index}",
        "Name": f"/harness-{index}",
        "Image": BASE_ID,
        "State": {"Running": True, "Health": {"Status": "healthy"}},
        "Config": {
            "Image": "goodvision-harness:installed",
            "User": "10001:10001",
            "Env": ["KEY=keep-secret", "HARNESS_FLEET_CAPACITY=2"],
            "Cmd": ["goodvision-harness"],
            "Labels": {
                "com.docker.compose.project": "goodvision-harness",
                "com.docker.compose.service": "harness",
            },
        },
        "HostConfig": {
            "Binds": ["data:/data:rw"],
            "RestartPolicy": {"Name": "unless-stopped"},
            "PortBindings": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "0"}]},
        },
        "Mounts": [
            {"Type": "volume", "Name": "data", "Destination": "/data", "RW": True},
            {
                "Type": "volume",
                "Name": "anonymous-model",
                "Destination": "/models",
                "RW": False,
            },
        ],
        "NetworkSettings": {
            "Ports": {
                "8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(8080 + index)}]
            },
            "Networks": {
                "fleet": {
                    "Aliases": [f"harness-{index}", f"old{index}"],
                    "IPAddress": f"172.16.0.{index}",
                }
            },
        },
    }


class FakeEngine:
    def __init__(self, *, fail=False, busy=False):
        self.containers = {f"old{i}": container(i) for i in (1, 2)}
        self.fail, self.busy = fail, busy
        self.calls = []
        self.created = []

    def inspect(self, identity):
        for value in self.containers.values():
            if identity in (value["Id"], value["Name"].lstrip("/")):
                return copy.deepcopy(value)
        raise RuntimeError("not found")

    def api(self, identity, path):
        if path.endswith("status"):
            return {"analysis_state": "running" if self.busy else "idle"}
        return {"autostart": False}

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        url = urlsplit(path)
        if path == "/containers/json?all=1":
            return [
                {
                    "Id": c["Id"],
                    "ImageID": c["Image"],
                    "Labels": c["Config"]["Labels"],
                    "Names": [c["Name"]],
                }
                for c in self.containers.values()
            ]
        if path.startswith("/images/"):
            return {"Id": "new-image"}
        if url.path == "/containers/create":
            if self.fail and len(self.created) == 1:
                raise RuntimeError("simulated creation failure")
            identity = "new" + str(len(self.created) + 1)
            self.created.append(body)
            self.containers[identity] = {
                "Id": identity,
                "Name": "/" + parse_qs(url.query)["name"][0],
                "Image": "new-image",
                "State": {"Running": False},
                "Config": body,
            }
            return {"Id": identity}
        parts = url.path.split("/")
        target = self.containers[parts[2]]
        if method == "DELETE":
            del self.containers[parts[2]]
        elif parts[3] == "stop":
            target["State"]["Running"] = False
        elif parts[3] == "start":
            target["State"]["Running"] = True
        elif parts[3] == "rename":
            name = "/" + parse_qs(url.query)["name"][0]
            assert not any(c["Name"] == name for c in self.containers.values())
            target["Name"] = name
        else:
            raise AssertionError(path)


MANIFEST = {
    "base_id": BASE_ID,
    "accepted_base_ids": [BASE_ID, *WINDOWS_BASE_IDS],
    "release": "test",
    "files": [],
}


def test_recreation_preserves_secrets_ports_networks_and_anonymous_models():
    original = container(1)
    before = copy.deepcopy(original)
    value = replacement_config(original, "updated")
    assert original == before
    assert value["Env"] == original["Config"]["Env"]
    assert value["HostConfig"]["Binds"] == ["data:/data:rw"]
    assert value["HostConfig"]["Mounts"] == [
        {
            "Type": "volume",
            "Source": "anonymous-model",
            "Target": "/models",
            "ReadOnly": True,
        }
    ]
    assert value["HostConfig"]["PortBindings"]["8080/tcp"][0]["HostPort"] == "8081"
    assert value["NetworkingConfig"]["EndpointsConfig"] == {
        "fleet": {"Aliases": ["harness-1"]}
    }


def test_realtime_release_migrates_old_flags_without_losing_credentials():
    original = container(1)
    original["Config"]["Env"] += ["LOCAL_CAMERA_TARGET_FPS=5", "SCENE_INDEPENDENT_VLM=false"]
    value = replacement_config(original, "updated", continuous_playback_vlm=True)
    environment = dict(item.split("=", 1) for item in value["Env"])
    assert environment["LOCAL_CAMERA_TARGET_FPS"] == "15"
    assert environment["SCENE_INDEPENDENT_VLM"] == "true"
    assert environment["SCENE_ENGINE_MODE"] == "active"
    assert environment["KEY"] == "keep-secret"
    assert environment["HARNESS_FLEET_CAPACITY"] == "2"


def test_release_contains_independent_vlm_and_browser_runtime_but_no_local_data(tmp_path):
    output = tmp_path / "update.zip"
    package(output)
    with zipfile.ZipFile(output) as archive:
        manifest_path = next(name for name in archive.namelist() if name.endswith("/manifest.json"))
        manifest = json.loads(archive.read(manifest_path))
        paths = {item["path"] for item in manifest["files"]}
    assert {"payload/vlm_independent.py", "payload/vlm_independent_model.py", "payload/result_bus.py", "payload/camera_display.js", "payload/assets/opencv-4.13.0.js"} <= paths
    assert not any(name.endswith((".sqlite3", ".pyc", ".mp4", ".onnx", ".env")) for name in paths)


def test_success_replaces_original_containers_without_retaining_backups(tmp_path):
    engine = FakeEngine()
    record = tmp_path / "state.json"
    upgrade(engine, MANIFEST, "updated", record)
    assert json.loads(record.read_text())["state"] == "complete"
    assert all(engine.containers[f"new{i}"]["State"]["Running"] for i in (1, 2))
    assert "keep-secret" not in record.read_text()
    assert set(engine.containers) == {"new1", "new2"}
    assert engine.containers["new1"]["Name"] == "/harness-1"
    assert not any(
        "rollback" in c["Name"] or "before-" in c["Name"]
        for c in engine.containers.values()
    )


def test_preparation_failure_leaves_original_services_untouched(tmp_path):
    engine = FakeEngine(fail=True)
    with pytest.raises(RuntimeError, match="simulated"):
        upgrade(engine, MANIFEST, "updated", tmp_path / "state.json")
    assert set(engine.containers) == {"old1", "old2"}
    for i in (1, 2):
        assert engine.containers[f"old{i}"] == container(i)


def test_busy_analysis_and_check_only_never_mutate_containers(tmp_path):
    for busy in (True, False):
        engine = FakeEngine(busy=busy)
        if busy:
            with pytest.raises(RuntimeError, match="Stop all analysis"):
                upgrade(engine, MANIFEST, "updated", tmp_path / "state.json")
        else:
            upgrade(
                engine, MANIFEST, "updated", tmp_path / "state.json", check_only=True
            )
        assert not any(method != "GET" for method, _, _ in engine.calls)
        assert not (tmp_path / "state.json").exists()


def test_temporary_helper_with_inherited_image_labels_is_excluded(tmp_path):
    engine = FakeEngine()
    helper = container(3)
    helper["Config"]["Labels"]["goodvision.update.helper"] = "true"
    engine.containers["old3"] = helper
    upgrade(engine, MANIFEST, "updated", tmp_path / "state.json")
    assert engine.containers["old3"] == helper
    assert len(engine.created) == 2


@pytest.mark.parametrize("base_id", WINDOWS_BASE_IDS)
def test_shipped_windows_project_and_secondary_service_are_supported(tmp_path, base_id):
    engine = FakeEngine()
    for i in (1, 2):
        c = engine.containers[f"old{i}"]
        c["Image"] = base_id
        c["Config"]["Labels"].update(
            {
                "com.docker.compose.project": "goodvision-harness-deploy",
                "com.docker.compose.service": "harness" if i == 1 else "harness-02",
            }
        )
    unrelated = container(3)
    engine.containers["old3"] = unrelated
    manifest = MANIFEST | {"accepted_base_ids": [BASE_ID, *WINDOWS_BASE_IDS]}
    record = tmp_path / "state.json"
    upgrade(
        engine,
        manifest,
        "updated",
        record,
        project="goodvision-harness-deploy",
        base_id=base_id,
    )
    assert len(engine.created) == 2
    assert engine.containers["old3"] == unrelated
    assert json.loads(record.read_text())["base_id"] == base_id


def test_unknown_secondary_version_is_rejected_before_any_mutation(tmp_path):
    engine = FakeEngine()
    engine.containers["old2"]["Image"] = "unknown"
    engine.containers["old2"]["Config"]["Labels"]["com.docker.compose.service"] = (
        "harness-02"
    )
    with pytest.raises(RuntimeError, match="versions differ"):
        upgrade(engine, MANIFEST, "updated", tmp_path / "state.json")
    assert not any(method != "GET" for method, _, _ in engine.calls)


def test_windows_launcher_accepts_update_and_keeps_archive_reference(tmp_path):
    import hashlib

    # The old multi-GB delivery can be removed without disabling updater tests.
    before = {
        "RELEASE.json": json.dumps({"accepted_image_ids": list(WINDOWS_BASE_IDS), "image": "original-image.tar"}).encode(),
        "SOURCE_HASHES.json": json.dumps({"image_id": WINDOWS_BASE_IDS[0], "source": "original", "files": {}}).encode(),
    }
    before["FILES.json"] = json.dumps({"files": [
        {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in before.items()
    ]}).encode()
    for name, data in before.items():
        (tmp_path / name).write_bytes(data)
    backup = read_metadata(tmp_path, WINDOWS_BASE_IDS[0])
    manifest = MANIFEST | {
        "files": [{"path": "payload/fleet_diagnostics.py", "sha256": "updated-source"}]
    }
    update_metadata(tmp_path, backup, manifest, "new-image")
    release = json.loads((tmp_path / "RELEASE.json").read_text())
    assert "new-image" in release["accepted_image_ids"]
    assert release["image"] == json.loads(before["RELEASE.json"])["image"]
    read_metadata(tmp_path, "new-image")
    assert (
        json.loads((tmp_path / "SOURCE_HASHES.json").read_text())["files"][
            "fleet_diagnostics.py"
        ]
        == "updated-source"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows entry points")
@pytest.mark.parametrize("entry", ["default", "explicit", "CHECK.cmd", "UPDATE.cmd"])
def test_native_windows_entry_points(tmp_path, entry):
    archive = tmp_path / "update.zip"
    package(archive)
    extracted = tmp_path / "package with spaces [test]"
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    bundle = next(extracted.iterdir())
    stub = tmp_path / "fake docker"
    stub.mkdir()
    installed = tmp_path / "original Windows install"
    installed.mkdir()
    (installed / "RELEASE.json").write_text("{}")
    targets = []
    for i in (1, 2):
        target = container(i)
        target["Image"] = WINDOWS_BASE_IDS[2]
        target["Config"]["Labels"].update(
            {
                "com.docker.compose.project": "goodvision-harness-deploy",
                "com.docker.compose.service": "harness" if i == 1 else "harness-02",
                "com.docker.compose.project.working_dir": str(installed),
            }
        )
        targets.append(target)
    (stub / "containers.json").write_text(json.dumps(targets), encoding="utf-8")
    (stub / "docker.cmd").write_text(
        r"""@echo off
>>"%~dp0calls.log" echo %*
if "%~1"=="context" goto context
if "%~1"=="info" (echo linux& exit /b 0)
if "%~1"=="ps" (echo old1& echo old2& exit /b 0)
if "%~1"=="inspect" (type "%~dp0containers.json"& exit /b 0)
if "%~1"=="image" (echo BASE_ID& exit /b 0)
if "%~1"=="run" (echo runner ok& exit /b 0)
if "%~1"=="build" exit /b 0
if "%~1"=="tag" exit /b 0
exit /b 91
:context
echo [{"Name":"desktop-linux","Endpoints":{"docker":{"Host":"npipe://fixture"}}}]
exit /b 0
""".replace("BASE_ID", WINDOWS_BASE_IDS[2]),
        encoding="ascii",
    )
    environment = os.environ | {"PATH": str(stub) + os.pathsep + os.environ["PATH"]}
    for key in list(environment):
        if key.upper() == "PSMODULEPATH":
            environment.pop(key)
    if entry.endswith(".cmd"):
        invocation = ["cmd.exe", "/d", "/c", str(bundle / entry)]
    else:
        invocation = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(bundle / "update_harness.ps1"),
            "-CheckOnly",
        ]
        if entry == "explicit":
            invocation += ["-PackageDir", str(bundle)]
    result = subprocess.run(
        invocation,
        env=environment,
        cwd=tmp_path,
        input="\n",
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = (stub / "calls.log").read_text(encoding="utf-8")
    assert "runner ok" in result.stdout
    assert "--project goodvision-harness-deploy" in calls
    assert BASE_ID not in calls
    assert ("build " in calls) == (entry == "UPDATE.cmd")
