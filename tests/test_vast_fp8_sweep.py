"""Offline load invariants, kernel provenance and live-server mismatch guards."""

import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools.vast_vlm import fp8_adapter, fp8_kernels, fp8_profile, fp8_sweep, runtime
from tools.vlm_benchmark import modes


@pytest.mark.parametrize("n", [16, 24, 32])
@pytest.mark.parametrize("admission", ["matched", "fixed48"])
def test_load_and_schedule_only_change_capacity(n, admission):
    exp = fp8_sweep.Experiment(n, admission)
    original = vars(fp8_adapter.arguments())
    args = fp8_sweep.arguments(exp)
    altered = {k for k, v in vars(args).items() if original.get(k) != v}
    assert altered <= {"global_limit", "fp8_experiment"}
    assert args.global_limit == (n if admission == "matched" else 48)
    assert args.per_stream_limit == args.local_concurrency == 4
    assert args.person_ids_only and args.record
    jobs = list(modes.jobs(args))
    assert len(jobs) == 6
    assert {j["streams"] for j in jobs} == {12}
    assert fp8_adapter.request_budget(args)["measurement_upper"] == 6624
    cmd = runtime.serve_command(Path("/models"), p=fp8_sweep.deployment_profile(exp))
    assert cmd[cmd.index("--max-num-seqs") + 1] == str(n)
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "2048"
    assert cmd[cmd.index("--moe-backend") + 1] == "auto"
    assert "--quantization" in cmd and "--cpu-offload-gb" not in cmd


@pytest.mark.parametrize(
    "field,value",
    [
        ("streams", [8]),
        ("observation_fps", 1),
        ("queue_timeout", 3),
        ("http_timeout", 3),
        ("local_concurrency", 2),
        ("request_rate", 15),
        ("person_ids_only", False),
        ("global_limit", 48),
        ("parallel_scheduling", "independent"),
        ("model", "other"),
    ],
)
def test_sweep_rejects_load_protocol_drift(field, value):
    args = fp8_sweep.arguments(fp8_sweep.Experiment(24))
    setattr(args, field, value)
    with pytest.raises(ValueError, match="frozen load"):
        modes.validate(args)


@pytest.mark.parametrize("action", ["plan", "start", "preflight", "run", "compare"])
def test_dry_run_never_accesses_runtime(action, monkeypatch, capsys):
    def forbidden(*a, **k):
        pytest.fail("dry run touched runtime/network")

    monkeypatch.setattr(runtime, "volume_root", forbidden)
    monkeypatch.setattr(fp8_sweep, "run", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    fp8_sweep.main([action, "--sequences", "24"])
    result = json.loads(capsys.readouterr().out)
    assert result["execute"] is False
    assert result["formal_rounds"] == 18
    assert [v["server_sequences"] for v in result["variants"]] == [16, 24, 32]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequences": 48},
        {"sequences": True},
        {"admission": "adaptive"},
        {"kernel": "deep_gemm"},
        {"kernel": "triton-tuned"},
        {"kernel": "auto", "tuned_dir": "/models/tuned"},
        {"kernel": "triton-tuned", "tuned_dir": "relative"},
    ],
)
def test_only_explicit_supported_arms(kwargs):
    with pytest.raises(ValueError):
        fp8_sweep.Experiment(**kwargs)


def kernel_bundle(path):
    path.mkdir()
    row = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    data = {
        "triton_version": "3.test",
        **{str(n): row for n in fp8_kernels.BATCH_SIZES},
    }
    runtime.write_json(path / fp8_kernels.CONFIG_NAME, data)
    runtime.write_json(
        path / "tuning.json",
        {
            **fp8_kernels.IDENTITY,
            "triton_version": "3.test",
            "config_sha256": runtime.sha(path / fp8_kernels.CONFIG_NAME),
        },
    )
    return path


def test_verified_kernel_bundle_and_mutation(tmp_path):
    path = kernel_bundle(tmp_path / "tuned")
    evidence = fp8_kernels.verify_bundle(path)
    assert evidence["revision"] == fp8_profile.MODEL_REVISION
    exp = fp8_sweep.Experiment(24, kernel="triton-tuned", tuned_dir=str(path))
    assert fp8_sweep.kernel_evidence(exp, tmp_path) == evidence
    (path / fp8_kernels.CONFIG_NAME).write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        fp8_kernels.verify_bundle(path)


def test_kernel_cannot_live_outside_volume(tmp_path):
    path = kernel_bundle(tmp_path / "tuned")
    exp = fp8_sweep.Experiment(kernel="triton-tuned", tuned_dir=str(path))
    with pytest.raises(ValueError, match="verified model volume"):
        fp8_sweep.kernel_evidence(exp, tmp_path / "other")


def test_tuned_kernel_triton_version_and_environment(tmp_path, monkeypatch):
    path = kernel_bundle(tmp_path / "tuned")
    profile = fp8_sweep.deployment_profile(
        fp8_sweep.Experiment(kernel="triton-tuned", tuned_dir=str(path))
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "wrong")
    with pytest.raises(ValueError, match="Triton version"):
        profile.server_environment(tmp_path)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "3.test")
    assert profile.server_environment(tmp_path) == {
        "VLLM_TUNED_CONFIG_FOLDER": str(path)
    }


@pytest.mark.parametrize(
    "kernel,logged,valid",
    [
        ("auto", "TRITON", True),
        ("marlin", "MARLIN", True),
        ("marlin", "TRITON", False),
        ("auto", "", False),
    ],
)
def test_actual_kernel_attestation(tmp_path, kernel, logged, valid):
    profile = fp8_sweep.deployment_profile(fp8_sweep.Experiment(kernel=kernel))
    runtime.write_json(tmp_path / "deployment.json", {})
    text = (
        f"Using {logged} Fp8 MoE backend out of potential backends: []"
        if logged
        else ""
    )
    (tmp_path / "server.log").write_text(text)
    if valid:
        profile.attest_ready(tmp_path)
        result = json.loads((tmp_path / "deployment.json").read_text())
        assert result["kernel_attestation"]["validated"]
        assert not result["kernel_attestation"]["capacity_validated"]
    else:
        with pytest.raises(ValueError):
            profile.attest_ready(tmp_path)


def test_tuned_arm_rejects_default_kernel_fallback(tmp_path):
    path = kernel_bundle(tmp_path / "tuned")
    profile = fp8_sweep.deployment_profile(
        fp8_sweep.Experiment(kernel="triton-tuned", tuned_dir=str(path))
    )
    runtime.write_json(tmp_path / "deployment.json", {})
    (tmp_path / "server.log").write_text(
        f"Using TRITON Fp8 MoE backend\nUsing configuration from {path}\n"
        "Using default MoE config. Performance might be sub-optimal!"
    )
    with pytest.raises(ValueError, match="no default fallback"):
        profile.attest_ready(tmp_path)


def test_old_live_server_cannot_be_relabelled(tmp_path, monkeypatch):
    exp = fp8_sweep.Experiment(24)
    profile = fp8_sweep.deployment_profile(exp)
    cmd = runtime.serve_command(tmp_path, p=profile)
    data = {
        "experiment": asdict(exp),
        "command": cmd,
        "kernel_bundle": None,
        "kernel_attestation": {"validated": True},
    }
    path = tmp_path / "deployment.json"
    runtime.write_json(path, data)
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: b"\0".join(s.encode() for s in cmd)
    )
    active = {"pid": 123, "deployment": str(path)}
    profile.verify_deployment(tmp_path, active)
    old = cmd.copy()
    old[old.index("--max-num-seqs") + 1] = "16"
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: b"\0".join(s.encode() for s in old)
    )
    with pytest.raises(ValueError, match="live server"):
        profile.verify_deployment(tmp_path, active)


@pytest.mark.parametrize("action", ["plan", "tune", "verify"])
def test_kernel_actions_are_offline_by_default(action, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "volume_root", lambda *a: pytest.fail("GPU work"))
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("download")
    )
    fp8_kernels.main([action])
    result = json.loads(capsys.readouterr().out)
    assert not result["execute"]
    assert {16, 24, 32} <= set(result["batch_sizes"])
    assert "--tune" in result["command"]


def test_tuner_refuses_busy_gpu_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "volume_root", lambda p: p)
    monkeypatch.setattr(runtime, "verify_model", lambda *a, **k: {})
    monkeypatch.setattr(
        "importlib.metadata.version", lambda name: fp8_profile.ENGINE_VERSION
    )
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "1234\n")
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("download")
    )
    with pytest.raises(ValueError, match="GPU is occupied"):
        fp8_kernels.tune(tmp_path, tmp_path / "tuned")
    assert not (tmp_path / "tuned").exists()


def test_legacy_runner_cannot_mislabel_experiment_server(tmp_path, monkeypatch):
    path = tmp_path / "deployment.json"
    runtime.write_json(path, {"experiment": asdict(fp8_sweep.Experiment(24))})
    monkeypatch.setattr(
        runtime, "verify_active", lambda *a, **k: {"deployment": str(path)}
    )
    monkeypatch.setattr(
        "tools.vast_vlm.yolo_runtime.verify_active",
        lambda *a: pytest.fail("must reject before readiness/inference"),
    )
    with pytest.raises(ValueError, match="use fp8_sweep"):
        fp8_adapter.run("video", tmp_path / "new-run", "packet", tmp_path)


def test_tuned_arm_requires_exact_loaded_file_and_rejects_symlink(tmp_path):
    path = kernel_bundle(tmp_path / "tuned")
    exp = fp8_sweep.Experiment(kernel="triton-tuned", tuned_dir=str(path))
    profile = fp8_sweep.deployment_profile(exp)
    runtime.write_json(tmp_path / "deployment.json", {})
    prefix = f"Using TRITON Fp8 MoE backend\nTuned config folder: {path}\n"
    (tmp_path / "server.log").write_text(prefix)
    with pytest.raises(ValueError, match="load unconfirmed"):
        profile.attest_ready(tmp_path)
    (tmp_path / "server.log").write_text(
        prefix
        + f"Using configuration from {path / fp8_kernels.CONFIG_NAME} for MoE layer."
    )
    profile.attest_ready(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(path, target_is_directory=True)
    exp = fp8_sweep.Experiment(kernel="triton-tuned", tuned_dir=str(alias))
    with pytest.raises(ValueError, match="symlink"):
        fp8_sweep.kernel_evidence(exp, tmp_path)


def mock_gpu_runtime(root, monkeypatch):
    gpu = SimpleNamespace(
        name="NVIDIA GeForce RTX 4090", total_memory=48 * 1024**3, major=8, minor=9
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: 1, get_device_properties=lambda n: gpu
            )
        ),
    )
    monkeypatch.setattr(runtime, "volume_root", lambda p: p)
    monkeypatch.setattr(runtime, "verify_model", lambda *a, **k: {})
    monkeypatch.setattr(
        runtime.importlib.metadata, "version", lambda name: fp8_profile.ENGINE_VERSION
    )
    monkeypatch.setattr(runtime, "boot_id", lambda: "test-boot")
    monkeypatch.setattr(runtime, "process_start_ticks", lambda pid: "10")
    monkeypatch.setattr(runtime, "wait_for_server", lambda process: 0.1)
    model = runtime.model_dir(root, p=fp8_profile)
    model.mkdir(parents=True)
    runtime.write_json(
        model / "config.json",
        {
            "text_config": {"num_experts": 256, "moe_intermediate_size": 512},
            "quantization_config": {"weight_block_size": [128, 128]},
        },
    )


@pytest.mark.parametrize("attestation_ok", [True, False])
def test_startup_applies_options_and_cleans_private_process_on_failure(
    tmp_path, monkeypatch, attestation_ok
):
    mock_gpu_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime.socket, "socket", MagicMock())
    process = SimpleNamespace(pid=12345, wait=lambda: 0)
    stopped = []
    monkeypatch.setattr(runtime, "stop_process_group", lambda p: stopped.append(p))
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", "/stray/old/config")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-forward")

    def launch(command, **kw):
        assert command[command.index("--max-num-seqs") + 1] == "32"
        assert command[command.index("--max-num-batched-tokens") + 1] == "2048"
        assert kw["start_new_session"] is True
        assert "VLLM_TUNED_CONFIG_FOLDER" not in kw["env"]
        assert "OPENROUTER_API_KEY" not in kw["env"]
        if attestation_ok:
            kw["stdout"].write("Using TRITON Fp8 MoE backend\n")
            kw["stdout"].flush()
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    profile = fp8_sweep.deployment_profile(fp8_sweep.Experiment(32))
    if attestation_ok:
        runtime.start(tmp_path, p=profile)
    else:
        with pytest.raises(ValueError, match="did not attest"):
            runtime.start(tmp_path, p=profile)
    assert stopped == [process]
    assert not (tmp_path / "runtime/active.json").exists()
    data = json.loads(
        next((tmp_path / "runtime").glob("*/deployment.json")).read_text()
    )
    assert data["experiment"]["sequences"] == 32
    assert bool(data.get("kernel_attestation")) is attestation_ok


@pytest.mark.parametrize("success", [True, False])
def test_tuning_persists_verified_bundle_only_after_success(
    tmp_path, monkeypatch, success
):
    mock_gpu_runtime(tmp_path, monkeypatch)
    template = kernel_bundle(tmp_path / "template")
    config = json.loads((template / fp8_kernels.CONFIG_NAME).read_text())
    output = tmp_path / "tuned"
    source = tmp_path / "tools/vllm-0.31.0/benchmark_moe.py"
    source.parent.mkdir(parents=True)
    source.write_text("# mocked pinned upstream tuner\n")
    monkeypatch.setattr(fp8_kernels, "SOURCE_SHA256", runtime.sha(source))
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("download")
    )
    process = SimpleNamespace(pid=54321, wait=lambda: 0 if success else 1)
    stopped = []
    monkeypatch.setattr(runtime, "stop_process_group", lambda p: stopped.append(p))

    def launch(cmd, **kw):
        assert "16" in cmd and "24" in cmd and "32" in cmd
        assert kw["cwd"] == output and kw["start_new_session"]
        assert kw["env"]["HF_HUB_OFFLINE"] == "1"
        assert "PYTHONPATH" not in kw["env"]
        runtime.write_json(output / fp8_kernels.CONFIG_NAME, config)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    if success:
        evidence = fp8_kernels.tune(tmp_path, output)
        assert evidence["config_sha256"] == runtime.sha(
            output / fp8_kernels.CONFIG_NAME
        )
        assert fp8_kernels.verify_bundle(output) == evidence
    else:
        with pytest.raises(RuntimeError, match="tuning failed"):
            fp8_kernels.tune(tmp_path, output)
        assert not (output / "tuning.json").exists()
    assert stopped == [process]


def test_adapter_inspector_is_spawn_picklable_and_preserves_transport(monkeypatch):
    inspector = pickle.loads(pickle.dumps(fp8_sweep.Adapter.inspector))
    config = object()
    monkeypatch.setattr(fp8_adapter, "inspector", lambda c, **kw: (c, kw))
    assert inspector(config, broker_url="local-broker") == (
        config,
        {"broker_url": "local-broker"},
    )


@pytest.mark.parametrize("preflight", [True, False])
def test_run_uses_selected_adapter_and_keeps_deployment_on_interruption(
    tmp_path, monkeypatch, preflight
):
    from tools.tracked_grounding import suite
    from tools.vast_vlm import benchmark, yolo_runtime

    path = tmp_path / "deployment.json"
    exp = fp8_sweep.Experiment(24)
    runtime.write_json(path, {"experiment": asdict(exp)})
    checked = []

    def verify(volume, *, p):
        checked.append(p.SERVER_SEQUENCES)
        return {"deployment": str(path)}

    monkeypatch.setattr(runtime, "verify_active", verify)
    monkeypatch.setattr(
        yolo_runtime, "verify_active", lambda *a: {"detector": "verified"}
    )
    monkeypatch.setattr(benchmark, "readiness", lambda: None)

    def run(video, output, detector, packet, *, model_adapter, preflight_only):
        assert checked == [24]
        assert model_adapter.arguments().global_limit == 24
        assert model_adapter.plan()["experiment"] == asdict(exp)
        assert preflight_only is preflight
        assert detector == fp8_profile.DETECTOR_URL
        output.mkdir()
        raise KeyboardInterrupt

    monkeypatch.setattr(suite, "run", run)
    output = tmp_path / "results"
    with pytest.raises(KeyboardInterrupt):
        fp8_sweep.run(
            exp, "video", output, "packet", tmp_path, preflight_only=preflight
        )
    assert json.loads((output / "deployment.json").read_text())["vlm"] == {
        "experiment": asdict(exp)
    }
