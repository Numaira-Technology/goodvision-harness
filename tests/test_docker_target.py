import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import deploy_modules as deploy
from tools import docker_modules as spec
from tools import docker_target as target


class Engine:
    def __init__(self, arch="aarch64", gpu=False, images=()):
        self.arch, self.gpu, self.images = arch, gpu, set(images)
        self.calls = []

    def __call__(self, command, **options):
        self.calls.append(command)
        if command[1] == "info":
            value = {"OSType": "linux", "Architecture": self.arch}
        elif command[1:3] == ["image", "inspect"]:
            if command[-1] not in self.images:
                raise subprocess.CalledProcessError(1, command)
            value = [{"Os": "linux", "Architecture": "amd64"}]
        elif command[1] == "run" and not self.gpu:
            raise subprocess.CalledProcessError(1, command)
        else:
            value = {}
        return SimpleNamespace(stdout=json.dumps(value), returncode=0)


def automatic():
    settings = spec.defaults("site", "v1")
    settings["bbox_device"] = "auto"
    return settings


@pytest.mark.parametrize("arch", ["aarch64", "arm64"])
def test_arm_engine_uses_native_cpu_without_starting_a_gpu_probe(arch):
    docker = Engine(arch=arch)
    selected = target.select_bbox(automatic(), docker)
    assert selected["platform"] == "linux/arm64" and selected["device"] == "cpu"
    assert len(docker.calls) == 1


@pytest.mark.parametrize("gpu,device", [(True, "cuda"), (False, "cpu")])
def test_amd_engine_probes_actual_container_gpu_and_cleans_up(gpu, device):
    docker = Engine("x86_64", gpu, ["goodvision-bbox:v1-cuda"])
    selected = target.select_bbox(automatic(), docker)
    assert selected["platform"] == "linux/amd64" and selected["device"] == device
    probe = next(command for command in docker.calls if command[1] == "run")
    assert "--pull" in probe and probe[probe.index("--pull") + 1] == "never"
    assert "torch.cuda.is_available()" in probe[-1]
    cleanup = docker.calls[-1]
    assert cleanup[:3] == ["docker", "rm", "-f"]
    assert cleanup[-1] == probe[probe.index("--name") + 1]
    assert not any("pull" == command[1] for command in docker.calls)


def test_explicit_cpu_skips_gpu_and_explicit_cuda_never_silently_falls_back():
    settings = automatic()
    settings["bbox_device"] = "cpu"
    docker = Engine("amd64", True)
    assert target.select_bbox(settings, docker)["device"] == "cpu"
    assert len(docker.calls) == 1
    settings["bbox_device"] = "cuda"
    with pytest.raises(ValueError, match="unavailable"):
        target.select_bbox(settings, Engine("amd64", False))
    with pytest.raises(ValueError, match="linux/amd64"):
        target.select_bbox(settings, Engine("arm64"))


def test_cross_build_does_not_require_gpu_on_the_build_machine():
    settings = automatic()
    settings["bbox_device"] = "cuda"
    docker = Engine("arm64")
    result = target.select_bbox(settings, docker, platform="linux/amd64", building=True)
    assert result["device"] == "cuda" and len(docker.calls) == 1


def test_engine_architecture_and_gpu_are_independent_of_launcher_os(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    result = target.select_bbox(
        automatic(), Engine("amd64", True, ["goodvision-bbox:v1-cuda"]),
    )
    assert result["platform"] == "linux/amd64" and result["device"] == "cuda"


def test_unsupported_engine_and_daemon_outage_fail_before_any_container():
    with pytest.raises(ValueError, match="Linux Docker"):
        target.select_bbox(automatic(), Engine("riscv64"))

    def unavailable(command, **options):
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        target.select_bbox(automatic(), unavailable)


def test_new_bbox_config_defaults_auto_but_plan_and_config_stay_offline(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(deploy, "location", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(deploy, "run", lambda *a, **kw: pytest.fail("Docker call"))
    state = tmp_path / "state"
    common = ["--modules", "bbox", "--state-dir", str(state)]
    assert deploy.main(["plan", *common]) == 0
    assert not state.exists()
    assert deploy.main(["configure", *common]) == 0
    saved = json.loads((state / "settings.json").read_text())
    assert saved["bbox_device"] == "auto" and "bbox_platform" not in saved


def test_auto_source_package_preserves_existing_explicit_device(tmp_path, monkeypatch):
    state = tmp_path / "state"
    settings = spec.defaults()
    settings["bbox_device"] = "cuda"
    deploy.write_json(state / "settings.json", settings)
    deploy.write_json(tmp_path / "module-package.json", {
        "module": "bbox", "version": "v2", "bbox_device": "auto", "files": {},
    })
    monkeypatch.setattr(deploy, "location", lambda: (tmp_path, tmp_path))
    assert deploy.main(["configure", "--state-dir", str(state)]) == 0
    assert json.loads((state / "settings.json").read_text())["bbox_device"] == "cuda"


def test_resolution_changes_only_bbox_and_uses_platform_qualified_image():
    settings = automatic()
    before = copy.deepcopy(settings)
    resolved, selected = target.resolve_bbox(settings, Engine())
    assert settings == before
    assert resolved["module_versions"] == settings["module_versions"]
    assert spec.bbox_image(resolved) == "goodvision-bbox:v1-cpu-arm64"
    service = spec.compose("bbox", resolved, Path("."), Path("."))["services"]["bbox"]
    assert service["platform"] == selected["platform"]
    assert "deploy" not in service
    assert spec.runtime_versions(["bbox"], resolved) == []


def test_first_source_install_falls_back_when_built_torch_cannot_run_gpu(
    tmp_path, monkeypatch,
):
    state = tmp_path / "state"
    resolutions, built = [], []

    def resolve(settings, run, **options):
        result = copy.deepcopy(settings)
        device = "cuda" if not resolutions else "cpu"
        result.update(bbox_platform="linux/amd64", bbox_runtime_device=device)
        resolutions.append(device)
        return result, {"device": device}

    def run(command, **options):
        if command[-1] == "build":
            document = json.loads((state / "compose.bbox.json").read_text())
            built.append(document["services"]["bbox"]["environment"]["DETECTOR_DEVICE"])

    monkeypatch.setattr(target, "resolve_bbox", resolve)
    monkeypatch.setattr(deploy, "location", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(deploy, "run", run)
    monkeypatch.setattr(deploy, "ensure_network", lambda _: None)
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0))
    assert deploy.main(["up", "--build", "--modules", "bbox",
                        "--state-dir", str(state)]) == 0
    assert resolutions == built == ["cuda", "cpu"]
    assert json.loads((state / "settings.json").read_text())["bbox_device"] == "auto"
    assert not (state / "compose.main.json").exists()
