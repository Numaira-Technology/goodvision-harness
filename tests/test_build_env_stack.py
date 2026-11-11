"""Offline integration of hardened builds with the CPU-only vLLM recipe."""

import base64
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from env_build_helpers import synthetic_pack

from tools import build_env as be
from tools import registry


def cpu_facts():
    return {
        "python": "3.12",
        "torch": "2.13.0+cu130",
        "cuda": "13.0",
        "compute_capabilities": [],
        "extensions": {"vllm": "0.31.0"},
    }


@pytest.fixture
def cpu_build(tmp_path, monkeypatch):
    spec = copy.deepcopy(registry.env_of("qwen8b-vllm"))
    spec.update(
        unpack_to=str(tmp_path / "prefix"),
        r2_key="envs/qwen8b-vllm/synthetic-successor.tar.gz",
        manifest_key="envs/qwen8b-vllm/synthetic-successor.json",
    )
    monkeypatch.setattr(registry, "env_of", lambda name: spec)
    monkeypatch.setattr(be.platform, "system", lambda: "Linux")
    monkeypatch.setattr(be.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        be.env_source,
        "check_compatibility",
        lambda *a: pytest.fail("CPU build probed GPU"),
    )
    monkeypatch.setattr(be, "_ensure_miniconda", lambda *a: tmp_path / "conda")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "create" in argv:
            Path(spec["unpack_to"]).mkdir()
            (Path(spec["unpack_to"]) / "keep").write_text("local builder state")
        return SimpleNamespace(stdout=json.dumps(cpu_facts()))

    monkeypatch.setattr(be, "_run", run)
    monkeypatch.setattr(be, "_pack", lambda conda, prefix, out: synthetic_pack(out))
    monkeypatch.setattr(be, "_verify_packed_build", lambda *a: None)
    monkeypatch.setattr(
        be.model_source, "r2_client", lambda: pytest.fail("unmocked R2 access")
    )
    out = tmp_path / "successor.tar.gz"
    args = ["--model", "qwen8b-vllm", "--out", str(out)]
    return spec, out, args, calls


def test_cpu_recipe_records_deferred_gpu_and_uploads_only_bound_bytes(
    cpu_build, monkeypatch, capsys
):
    spec, out, args, calls = cpu_build
    be.main([*args, "--skip-upload"])
    record = json.loads(be._record_path(out).read_text())
    assert record["schema_version"] == 2
    assert record["gpu_validation"] == "deferred"
    assert record["runtime"] == cpu_facts()
    assert record["build_options"] == be._build_options(spec)
    recipe = be._requirements("qwen8b-vllm")
    assert recipe == Path(be.__file__).parent / "qwen8b_vllm/requirements.txt"
    assert any(str(recipe) in call for call in calls)
    assert not any("torch==2.8.0" in call or "setuptools" in call for call in calls)
    puts = []

    class R2:
        def put_object(self, **kwargs):
            body = kwargs["Body"]
            puts.append(
                {**kwargs, "Body": body.read() if hasattr(body, "read") else body}
            )

    monkeypatch.setattr(be.model_source, "r2_client", R2)
    monkeypatch.setenv("R2_BUCKET", "synthetic")
    calls.clear()
    be.main([*args, "--upload-only"])
    assert not calls  # upload host never executes the artifact or probes CUDA
    assert len(puts) == 2 and all(p["IfNoneMatch"] == "*" for p in puts)
    manifest = json.loads(puts[1]["Body"])
    assert manifest["build"] == record
    assert manifest["r2_key"] == f"envs/qwen8b-vllm/{be._sha256(out)}.tar.gz"
    assert manifest["r2_key"] not in be.FROZEN_R2_KEYS


@pytest.mark.parametrize("fault", ["gpu-claim", "caps", "cuda", "vllm", "options"])
def test_cpu_upload_rejects_changed_evidence(cpu_build, fault):
    _spec, out, args, _calls = cpu_build
    be.main([*args, "--skip-upload"])
    path = be._record_path(out)
    record = json.loads(path.read_text())
    if fault == "gpu-claim":
        record["gpu_validation"] = "measured"
    elif fault == "caps":
        record["runtime"]["compute_capabilities"] = ["8.9"]
    elif fault == "options":
        record["build_options"]["require_cuda"] = 0
    elif fault == "cuda":
        record["runtime"]["cuda"] = "12.8"
    else:
        record["runtime"]["extensions"]["vllm"] = "0.33.0"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        be.main([*args, "--upload-only"])


def test_staged_build_preserves_prefix_and_reverifies_before_pack(cpu_build):
    spec, out, args, calls = cpu_build
    be.main([*args, "--skip-pack"])
    stage = be._prefix_record_path(out)
    original = stage.read_bytes()
    assert not out.exists() and not be._record_path(out).exists()
    marker = Path(spec["unpack_to"]) / "keep"
    calls.clear()
    be.main([*args, "--pack-only", "--skip-upload"])
    assert out.is_file() and be._record_path(out).is_file()
    assert stage.read_bytes() == original
    assert json.loads(be._record_path(out).read_text())["prefix_build"] == json.loads(
        original
    )
    assert marker.read_text() == "local builder state"
    assert len(calls) == 1 and calls[0][1:3] == ["-I", "-c"]


@pytest.mark.parametrize("fault", ["missing", "recipe", "name", "runtime", "symlink"])
def test_pack_only_requires_matching_trusted_prefix_record(cpu_build, fault):
    spec, out, args, calls = cpu_build
    be.main([*args, "--skip-pack"])
    path = be._prefix_record_path(out)
    if fault == "missing":
        path.unlink()
    elif fault == "symlink":
        path.rename(path.with_suffix(".saved"))
        path.symlink_to(path.with_suffix(".saved"))
    else:
        record = json.loads(path.read_text())
        if fault == "recipe":
            record["requirements_sha256"] = "0" * 64
        elif fault == "name":
            record["name"] = "qwen8b"
        else:
            record["runtime"]["torch"] = "2.8.0+cu128"
        path.write_text(json.dumps(record))
    calls.clear()
    with pytest.raises(ValueError):
        be.main([*args, "--pack-only", "--skip-upload"])
    assert not calls and not out.exists()
    assert Path(spec["unpack_to"]).is_dir()


@pytest.mark.parametrize(
    "flags",
    [
        ["--skip-pack", "--pack-only"],
        ["--skip-pack", "--upload-only"],
        ["--pack-only", "--upload-only"],
    ],
)
def test_stage_modes_are_mutually_exclusive(cpu_build, flags):
    _spec, out, args, calls = cpu_build
    with pytest.raises(SystemExit):
        be.main([*args, *flags])
    assert not calls and not out.exists()


@pytest.mark.parametrize(
    "field", ["require_cuda", "install_default_torch", "flash_attn"]
)
def test_build_options_do_not_coerce_false_strings(field):
    spec = copy.deepcopy(registry.env_of("qwen8b-vllm"))
    spec["build"][field] = "false"
    with pytest.raises(ValueError, match="boolean"):
        be._build_options(spec)


def test_restore_verification_checks_safe_fresh_prefix(tmp_path, monkeypatch):
    env = registry.env_of("qwen8b-vllm")
    out = synthetic_pack(tmp_path / "pack.tar.gz")
    restored_paths = []

    def unpack(prefix):
        restored_paths.append(prefix)
        assert (prefix / "bin/python").read_bytes().startswith(b"synthetic")

    monkeypatch.setattr(be.env_source, "_run_unpack", unpack)
    monkeypatch.setattr(be, "_probe_prefix", lambda *a: cpu_facts())
    be._verify_packed_build(out, env, cpu_facts())
    assert len(restored_paths) == 1 and not restored_paths[0].exists()
    wrong = cpu_facts()
    wrong["torch"] = "unexpected"
    with pytest.raises(ValueError, match="restored runtime"):
        be._verify_packed_build(out, env, wrong)


def test_missing_conda_files_fail_before_pack(cpu_build):
    spec, out, args, _calls = cpu_build
    be.main([*args, "--skip-pack"])
    metadata = Path(spec["unpack_to"]) / "conda-meta"
    metadata.mkdir()
    (metadata / "setuptools.json").write_text(
        json.dumps(
            {
                "name": "setuptools",
                "version": "84.2.1",
                "files": ["lib/python3.12/site-packages/setuptools/missing.py"],
            }
        )
    )
    with pytest.raises(RuntimeError, match="conda-meta lists files missing"):
        be._assert_prefix_files(Path(spec["unpack_to"]))
    with pytest.raises(ValueError, match="prefix contents changed"):
        be.main([*args, "--pack-only", "--skip-upload"])
    assert not out.exists()


def test_generated_cpu_probe_runs_without_any_gpu_call():
    """Execute the generated verifier against in-memory synthetic modules only."""
    env = registry.env_of("qwen8b-vllm")
    prelude = """
import sys, types, importlib.metadata
torch = types.ModuleType('torch')
torch.__file__ = sys.prefix + '/synthetic/torch.py'
torch.__version__ = '2.13.0+cu130'
torch.version = types.SimpleNamespace(cuda='13.0')
class NoCuda:
    def __getattr__(self, name):
        raise AssertionError('unexpected GPU call: ' + name)
torch.cuda = NoCuda()
sys.modules['torch'] = torch
for name in ['typing_extensions', 'vllm']:
    mod = types.ModuleType(name)
    mod.__file__ = sys.prefix + '/synthetic/' + name + '.py'
    sys.modules[name] = mod
importlib.metadata.version = lambda name: '0.31.0'
real_import = importlib.import_module
def noisy_import(name):
    print('synthetic dependency import message')
    return real_import(name)
importlib.import_module = noisy_import
"""
    # The test interpreter can differ from the Linux builder's pinned Python.
    compat = {
        **env["compat"],
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    code = be._verify_code(False, compat, False, ["vllm"])
    result = subprocess.run(
        [sys.executable, "-I", "-c", prelude + code],
        check=True,
        capture_output=True,
        text=True,
    )
    facts = json.loads(result.stdout)
    assert facts == {**cpu_facts(), "python": compat["python"]}
    assert "synthetic dependency import message" in result.stderr


@pytest.mark.parametrize(
    "fault", ["content", "mode", "link", "escape", "added", "removed"]
)
def test_staged_prefix_changes_stop_before_execution(cpu_build, fault):
    spec, out, args, calls = cpu_build
    be.main([*args, "--skip-pack"])
    prefix = Path(spec["unpack_to"])
    keep = prefix / "keep"
    if fault == "content":
        keep.write_text("changed after the recorded build")
    elif fault == "mode":
        keep.chmod(0o755)
    elif fault in {"link", "escape"}:
        keep.unlink()
        keep.symlink_to("." if fault == "link" else "../outside")
    elif fault == "added":
        (prefix / "extra.py").write_text("UNRECORDED = True")
    else:
        keep.unlink()
    calls.clear()
    with pytest.raises(ValueError, match=r"prefix contents changed|unsafe link"):
        be.main([*args, "--pack-only", "--skip-upload"])
    assert calls == [] and not out.exists()


def test_staged_pack_retains_original_builder_identity(cpu_build):
    _spec, out, args, _calls = cpu_build
    be.main([*args, "--skip-pack"])
    path = be._prefix_record_path(out)
    record = json.loads(path.read_text())
    record["builder_sha256"] = "1" * 64  # synthetic earlier trusted producer
    path.write_text(json.dumps(record))
    be.main([*args, "--pack-only", "--skip-upload"])
    packed = json.loads(be._record_path(out).read_text())
    assert packed["prefix_build"] == record
    assert packed["builder_sha256"] == be._sha256(Path(be.__file__))


def test_staged_prefix_change_during_pack_never_creates_receipt(cpu_build, monkeypatch):
    spec, out, args, _calls = cpu_build
    be.main([*args, "--skip-pack"])

    def pack(conda, prefix, target):
        synthetic_pack(target)
        (Path(spec["unpack_to"]) / "keep").write_text("changed during packing")

    monkeypatch.setattr(be, "_pack", pack)
    with pytest.raises(ValueError, match="during packing"):
        be.main([*args, "--pack-only", "--skip-upload"])
    assert out.exists() and not be._record_path(out).exists()


def test_prefix_fingerprint_binds_hardlink_relationship(tmp_path):
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    a, b = prefix / "a", prefix / "b"
    a.write_bytes(b"same bytes")
    b.write_bytes(a.read_bytes())
    before = be._prefix_fingerprint(prefix)
    b.unlink()
    b.hardlink_to(a)
    assert be._prefix_fingerprint(prefix) != before


def test_unreadable_staged_subtree_stops_before_execution(cpu_build, monkeypatch):
    spec, out, args, calls = cpu_build
    be.main([*args, "--skip-pack"])
    prefix = Path(spec["unpack_to"])
    real_scandir = be.os.scandir

    def denied(path):
        if Path(path) == prefix:
            raise PermissionError("synthetic traversal failure")
        return real_scandir(path)

    monkeypatch.setattr(be.os, "scandir", denied)
    calls.clear()
    with pytest.raises(ValueError, match="complete build prefix"):
        be.main([*args, "--pack-only", "--skip-upload"])
    assert calls == [] and not out.exists()


def test_source_record_hashes_allow_only_attested_conda_relocation(tmp_path):
    prefix = tmp_path / "prefix"
    site = prefix / "lib/python3.12/site-packages"
    dist = site / "synthetic.dist-info"
    dist.mkdir(parents=True)
    code = site / "example.py"
    original, installed = b"PREFIX = '/build/placeholder'\n", b"PREFIX = '/installed'\n"
    code.write_bytes(installed)
    old_sha = hashlib.sha256(original).hexdigest()
    digest = base64.urlsafe_b64encode(bytes.fromhex(old_sha)).decode().rstrip("=")
    (dist / "RECORD").write_text(f"example.py,sha256={digest},{len(original)}\n")
    with pytest.raises(RuntimeError, match="RECORD hash mismatch"):
        be._assert_prefix_files(prefix)
    metadata = prefix / "conda-meta"
    metadata.mkdir()
    rel = code.relative_to(prefix).as_posix()
    (metadata / "synthetic.json").write_text(
        json.dumps(
            {
                "name": "synthetic",
                "version": "1",
                "files": [rel],
                "paths_data": {
                    "paths": [
                        {
                            "_path": rel,
                            "file_mode": "text",
                            "prefix_placeholder": "/build/placeholder",
                            "sha256": old_sha,
                            "sha256_in_prefix": be._sha256(code),
                        }
                    ]
                },
            }
        )
    )
    assert be._assert_prefix_files(prefix)["record_relocations_verified"] == 1
    code.write_bytes(b"arbitrary replacement\n")
    with pytest.raises(RuntimeError, match="RECORD hash mismatch"):
        be._assert_prefix_files(prefix)


def test_legacy_gpu_build_record_still_validates(tmp_path):
    from env_build_helpers import runtime_facts

    env = registry.env_of("videojev")
    tarball = synthetic_pack(tmp_path / "gpu.tar.gz")
    recipe = be._sha256(be._requirements("videojev"))
    record = be._new_record(
        env, "videojev", tarball, "a/b@old", recipe, runtime_facts()
    )
    record["schema_version"] = 1
    del record["build_options"], record["gpu_validation"]
    be._validate_record(record, env, "videojev", tarball, "a/b@old", recipe)


def test_vllm_registry_keeps_active_and_historical_identities():
    env = registry.env_of("qwen8b-vllm")
    assert env["r2_key"] == "envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz"
    assert (
        env["sha256"]
        == "59314d00783b6698ef83cbfbd2e028c2f094d76c9d3383f5a37c3fdf3687b95b"
    )
    assert env["size"] == 4252375781
    assert env["compat"]["cuda"] == "13.0"
    assert (
        env["superseded"]["sha256"]
        == "2a3dcb1fc6a8d84439263512c65dae0bb576c50d4f559ae9f069407eb2e86db6"
    )
    assert (
        env["incomplete_cu130"]["sha256"]
        == "24f1eb0cc264a9794979eb6c2225eca47d67d68025407d51db86aa171ce7ba7e"
    )
    assert (
        "vllm.entrypoints.openai.api_server" in be._build_options(env)["verify_modules"]
    )
