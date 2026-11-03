import copy
import io
import json
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_module_updates import fixture, oci_archive, replace_archive

from tools import deploy_modules as deploy
from tools import docker_modules as spec
from tools import module_updates as updates


@pytest.mark.parametrize("first_action", ["success", "failure", "rollback"])
def test_interleaved_modules_preserve_each_others_versions(tmp_path, first_action):
    state, first, _, path, _, docker = fixture(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    _, _, bbox_doc, bbox_path, _, bbox_docker = fixture(other_root, "bbox")
    deploy.write_json(state / "compose.bbox.json", bbox_doc)
    docker.images.update(bbox_docker.images)
    if first_action == "rollback":
        updates.update(state, path, first, 30, docker, deploy.write_json)
    stale = copy.deepcopy(first)
    entered, release, second_entered = (threading.Event() for _ in range(3))

    def first_run(command, **options):
        if command[1] == "compose" and "up" in command:
            entered.set()
            assert release.wait(5)
        return docker(command, **options)

    def second_run(command, **options):
        second_entered.set()
        return bbox_docker(command, **options)

    def first_update():
        if first_action == "failure":
            docker.fail_pending = True
            with pytest.raises(ValueError, match="previous module was restored"):
                updates.update(state, path, first, 30, first_run, deploy.write_json)
        elif first_action == "rollback":
            updates.rollback(state, "main", first, 30, first_run, deploy.write_json)
        else:
            updates.update(state, path, first, 30, first_run, deploy.write_json)

    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(first_update)
        assert entered.wait(5)
        b = pool.submit(updates.update, state, bbox_path, stale, 30,
                        second_run, deploy.write_json)
        try:
            assert not second_entered.wait(.1)
        finally:
            release.set()
        a.result(5)
        b.result(5)
    saved = json.loads((state / "settings.json").read_text())
    assert saved["module_versions"]["bbox"] == "v2"
    assert saved["module_versions"]["main"] == (
        "v2" if first_action == "success" else "v1"
    )
    assert saved["module_images"]["bbox"]["bbox"] == "goodvision-bbox:v2-cpu"


def test_cli_waits_for_other_process_before_reading_shared_settings(tmp_path):
    state, _, _, _, _, _ = fixture(tmp_path)
    ready = tmp_path / "ready"
    script = '''
import sys
from pathlib import Path
from tools import deploy_modules as deploy
Path(sys.argv[1]).write_text("started")
raise SystemExit(deploy.main(["configure", "--modules", "bbox", "--namespace",
    "site", "--state-dir", sys.argv[2]]))
'''
    with spec.state_lock(state):
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(ready), str(state)],
                                   cwd=Path(__file__).resolve().parents[1],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            # Wait for the child's own startup marker, not an assumed scheduler delay.
            import time
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert ready.exists()
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(.2)
            saved = json.loads((state / "settings.json").read_text())
            saved["module_versions"]["main"] = "concurrent"
            deploy.write_json(state / "settings.json", saved)
        except BaseException:
            process.kill()
            process.communicate()
            raise
    out, err = process.communicate(timeout=10)
    assert process.returncode == 0, (out, err)
    saved = json.loads((state / "settings.json").read_text())
    assert saved["module_versions"]["main"] == "concurrent"


@pytest.mark.parametrize("extra", ["old", "other", "duplicate", "wrong_digest"])
def test_archive_aliases_or_identity_mismatch_rejected_before_import(tmp_path, extra):
    state, settings, _, path, manifest, docker = fixture(tmp_path)
    images = copy.deepcopy(manifest["images"])
    aliases = []
    if extra == "wrong_digest":
        images["main"]["id"] = "sha256:" + "f" * 64
    else:
        alias = dict(images["main"])
        if extra == "old":
            alias["name"] = "goodvision-main:v1"
        elif extra == "other":
            alias["name"] = "goodvision-bbox:v1-cpu"
        aliases.append(alias)
    replace_archive(path, manifest, oci_archive(images, extra=aliases))
    before = (state / "settings.json").read_bytes()
    with pytest.raises(ValueError, match="OCI"):
        updates.update(state, path, settings, 30, docker, deploy.write_json)
    assert docker.calls == []
    assert (state / "settings.json").read_bytes() == before
    assert not (state / "rollback.main.json").exists()


def test_previous_image_identity_captured_before_import_and_rechecked(tmp_path):
    state, settings, _, path, _, docker = fixture(tmp_path)

    def changed_during_import(command, **options):
        result = docker(command, **options)
        if command[1:3] == ["image", "load"]:
            assert any(call[1:3] == ["image", "inspect"]
                       and call[-1] == "goodvision-main:v1" for call in docker.calls)
            docker.images["goodvision-main:v1"] = "sha256:" + "f" * 64
        return result

    with pytest.raises(ValueError, match="Required image differs"):
        updates.update(
            state, path, settings, 30, changed_during_import, deploy.write_json,
        )
    assert not any(call[1] == "compose" for call in docker.calls)
    assert not (state / "rollback.main.json").exists()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_variant_switch_executes_real_health_contract_in_both_directions(
    tmp_path, monkeypatch, device,
):
    from test_bbox_variant_updates import release

    state, settings, old, path, _, run, _, _ = release(
        tmp_path, "amd64", device == "cuda",
    )
    previous = "cuda" if device == "cpu" else "cpu"
    old["services"]["bbox"]["healthcheck"] = spec.compose(
        "bbox", {**settings, "bbox_device": previous}, tmp_path, state,
    )["services"]["bbox"]["healthcheck"]
    deploy.write_json(state / "compose.bbox.json", old)
    checks = []

    def healthy_run(command, **options):
        if command[1] == "compose" and "up" in command:
            doc = json.loads(Path(command[command.index("-f") + 1]).read_text())
            check = doc["services"]["bbox"]["healthcheck"]["test"][-1]
            response = {"ready": True, "model": "yolo26-s", "device": device,
                        "runtime": "ultralytics", "provider_ready": True,
                        "gpu_ready": device == "cuda"}
            with monkeypatch.context() as patch:
                patch.setattr(urllib.request, "urlopen", lambda *a, **k:
                              io.BytesIO(json.dumps(response).encode()))
                exec(check, {})
                for broken in [{"ready": False}, {"model": "wrong"},
                               {"gpu_ready": device != "cuda"}]:
                    bad = {**response, **broken}
                    patch.setattr(urllib.request, "urlopen", lambda *a, bad=bad, **k:
                                  io.BytesIO(json.dumps(bad).encode()))
                    with pytest.raises(AssertionError):
                        exec(check, {})
            checks.append(check)
        return run(command, **options)

    updates.update(state, path, settings, 30, healthy_run, deploy.write_json)
    assert len(checks) == 1
