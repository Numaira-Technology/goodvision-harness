"""Offline end-to-end build-record tests; every build and storage effect is fake."""

import copy
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from env_build_helpers import completed_probe, runtime_facts, synthetic_pack

from tools import build_env as be
from tools import registry

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX packed-env prefixes")


@pytest.fixture
def fake_builder(tmp_path, monkeypatch, capsys):
    specs = {
        name: copy.deepcopy(registry.env_of(name)) for name in registry.model_names()
    }
    for name, spec in specs.items():
        spec["unpack_to"] = str(tmp_path / "prefixes" / name)
        # Publish a proposed successor, never a frozen historical identity.
        spec["r2_key"] = f"envs/{name}/synthetic-successor.tar.gz"
        spec["manifest_key"] = f"envs/{name}/synthetic-successor.json"
    monkeypatch.setattr(registry, "env_of", lambda name: specs[name])
    monkeypatch.setattr(be.env_source, "check_compatibility", lambda env: None)
    monkeypatch.setattr(be, "_ensure_miniconda", lambda *args: tmp_path / "conda")
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return completed_probe("import sys, torch, flash_attn" in argv[-1])

    monkeypatch.setattr(be, "_run", run)
    monkeypatch.setattr(be, "_pack", lambda conda, prefix, out: synthetic_pack(out))
    monkeypatch.setattr(be, "_verify_packed_build", lambda *args: None)

    def build(name="qwen8b", *, requirements=None):
        out = tmp_path / f"{name}.tar.gz"
        argv = ["--model", name, "--skip-upload", "--out", str(out)]
        if requirements is not None:
            argv.extend(["--requirements", str(requirements)])
        be.main(argv)
        result = json.loads(capsys.readouterr().out)
        record = json.loads(be._record_path(out).read_text())
        assert result["uploaded"] is False
        assert result["sha256"] == record["sha256"]
        assert result["build_record"] == str(be._record_path(out))
        return out, record

    return build, calls


@pytest.mark.parametrize(
    "name,directory,flash",
    [
        ("videojev", "videojev_probe", True),
        ("clip", "clip", False),
        ("siglip2", "siglip2", False),
        ("d3-nano", "d3_nano", True),
        ("qwen8b", "qwen8b", True),
    ],
)
def test_main_builds_selected_recipe_and_records_probe(
    fake_builder, name, directory, flash
):
    build, calls = fake_builder
    _out, record = build(name)
    recipe = Path(be.__file__).parent / directory / "requirements.txt"
    installed = next(call[call.index("-r") + 1] for call in calls if "-r" in call)
    assert installed == str(recipe.resolve())
    assert record["requirements_sha256"] == be._sha256(recipe)
    assert record["runtime"] == runtime_facts(flash)
    assert record["for_model"] == "@".join(registry.weights_of(name))
    assert any("flash-attn==2.8.3.post2" in c for c in calls) is flash


def test_explicit_recipe_and_record_can_move_to_cpu_upload_host(
    fake_builder, tmp_path, monkeypatch, capsys
):
    build, calls = fake_builder
    recipe = tmp_path / "custom.txt"
    recipe.write_text("numpy==2.2.6\n")
    out, record = build(requirements=recipe)
    assert any(str(recipe.resolve()) in c for c in calls)
    moved = tmp_path / "moved.tar.gz"
    out.rename(moved)
    be._record_path(out).rename(be._record_path(moved))
    uploads = []

    class FakeR2:
        def put_object(self, **kw):
            body = kw["Body"]
            uploads.append(
                {**kw, "Body": body.read() if hasattr(body, "read") else body}
            )

    monkeypatch.setattr(be.model_source, "r2_client", FakeR2)
    monkeypatch.setenv("R2_BUCKET", "synthetic")
    for function in ("_build_prefix", "_pack", "_ensure_miniconda"):
        monkeypatch.setattr(
            be, function, lambda *a: pytest.fail("upload tried to build")
        )
    monkeypatch.setattr(
        be.env_source,
        "check_compatibility",
        lambda *a: pytest.fail("upload tried to probe local GPU"),
    )
    be.main(
        [
            "--model",
            "qwen8b",
            "--out",
            str(moved),
            "--upload-only",
            "--requirements",
            str(recipe),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["uploaded"] is True
    assert result["build_record"] == str(be._record_path(moved))
    manifest = json.loads(uploads[1]["Body"])
    assert manifest["build"] == record
    assert manifest["sha256"] == be._sha256(moved)
    assert all(put["IfNoneMatch"] == "*" for put in uploads)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-record",
        "invalid-json",
        "oversize-record",
        "non-mapping",
        "schema",
        "unknown-field",
        "name",
        "revision",
        "prefix",
        "compat",
        "toggle",
        "builder",
        "recipe",
        "digest",
        "size",
        "runtime",
        "python",
        "torch",
        "cuda",
        "gpu",
        "extensions",
        "changed-bytes",
        "not-archive",
        "incomplete-archive",
        "other-model",
    ],
)
def test_upload_only_rejects_unbound_artifact_before_client(
    fake_builder, monkeypatch, fault
):
    build, _ = fake_builder
    out, record = build()
    model = "qwen8b"
    record_path = be._record_path(out)
    if fault == "missing-record":
        record_path.unlink()
    elif fault == "invalid-json":
        record_path.write_text("{")
    elif fault == "oversize-record":
        record_path.write_text(" " * (be.MAX_BUILD_RECORD_SIZE + 1))
    elif fault == "non-mapping":
        record_path.write_text("[]")
    else:
        if fault == "schema":
            record["schema_version"] = True
        elif fault == "unknown-field":
            record["unreviewed_extra"] = "unexpected"
        elif fault == "name":
            record["name"] = "clip"
        elif fault == "revision":
            record["for_model"] = "Qwen/Qwen3-VL-8B-Instruct@" + "0" * 40
        elif fault == "prefix":
            record["unpack_to"] = "/opt/env/other"
        elif fault == "compat":
            record["compat"]["gpu_family"] = "ampere"
        elif fault == "toggle":
            record["flash_attn"] = "true"
        elif fault == "builder":
            record.pop("builder_sha256")
        elif fault == "recipe":
            record["requirements_sha256"] = "0" * 64
        elif fault == "digest":
            record["sha256"] = "0" * 64
        elif fault == "size":
            record["size"] += 1
        elif fault == "runtime":
            record.pop("runtime")
        elif fault in {"python", "torch", "cuda"}:
            record["runtime"][fault] = "wrong"
        elif fault == "gpu":
            record["runtime"]["compute_capabilities"] = ["8.0"]
        elif fault == "extensions":
            record["runtime"]["extensions"] = {}
        elif fault == "changed-bytes":
            data = out.read_bytes()
            out.write_bytes(b"x" + data[1:])
        elif fault in {"not-archive", "incomplete-archive"}:
            if fault == "not-archive":
                out.write_bytes(b"not a conda pack")
            else:
                synthetic_pack(out, omit="bin/conda-unpack")
            record.update(sha256=be._sha256(out), size=out.stat().st_size)
        elif fault == "other-model":
            model = "clip"
        record_path.write_text(json.dumps(record))
    monkeypatch.setattr(
        be.model_source, "r2_client", lambda: pytest.fail("invalid artifact reached R2")
    )
    with pytest.raises((ValueError, be.tarfile.TarError)):
        be.main(["--model", model, "--out", str(out), "--upload-only"])


def test_changed_override_recipe_is_rejected(fake_builder, tmp_path, monkeypatch):
    build, _ = fake_builder
    recipe = tmp_path / "custom.txt"
    recipe.write_text("numpy==2.2.6\n")
    out, _record = build(requirements=recipe)
    recipe.write_text("numpy==1.26.4\n")
    monkeypatch.setattr(be.model_source, "r2_client", lambda: pytest.fail("R2 client"))
    with pytest.raises(ValueError, match="recipe"):
        be.main(
            [
                "--model",
                "qwen8b",
                "--out",
                str(out),
                "--upload-only",
                "--requirements",
                str(recipe),
            ]
        )


@pytest.mark.parametrize("stage", ["_build_prefix", "_pack"])
def test_failed_build_never_creates_upload_record(
    fake_builder, tmp_path, monkeypatch, stage
):
    build, _ = fake_builder

    def fail(*args):
        raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(be, stage, fail)
    with pytest.raises(RuntimeError, match="synthetic build failure"):
        build()
    assert not be._record_path(tmp_path / "qwen8b.tar.gz").exists()


def test_recipe_change_during_build_has_no_record(fake_builder, tmp_path, monkeypatch):
    build, _ = fake_builder
    recipe = tmp_path / "custom.txt"
    recipe.write_text("numpy==2.2.6\n")

    def pack(conda, prefix, out):
        synthetic_pack(out)
        recipe.write_text("numpy==1.26.4\n")

    monkeypatch.setattr(be, "_pack", pack)
    with pytest.raises(ValueError, match="requirements changed"):
        build(requirements=recipe)
    assert not be._record_path(tmp_path / "qwen8b.tar.gz").exists()


def test_existing_record_is_preserved_before_build(fake_builder, tmp_path):
    build, calls = fake_builder
    path = be._record_path(tmp_path / "qwen8b.tar.gz")
    path.write_text("preserve previous build evidence")
    with pytest.raises(SystemExit):
        build()
    assert path.read_text() == "preserve previous build evidence"
    assert not calls


@pytest.mark.parametrize("target", ["python3.12", "../../outside", "python"])
def test_archive_preflight_handles_python_symlink_without_execution(tmp_path, target):
    path = tmp_path / "symlink.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in ("bin/python3.12", "bin/activate", "bin/conda-unpack"):
            member = tarfile.TarInfo(name)
            data = b"synthetic file, never execute"
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))
        alias = tarfile.TarInfo("bin/python")
        alias.type, alias.linkname = tarfile.SYMTYPE, target
        archive.addfile(alias)
    if target == "python3.12":
        be._check_pack(path)
    else:
        with pytest.raises(ValueError):
            be._check_pack(path)


@pytest.mark.parametrize("bad_parent", [False, True])
def test_archive_preflight_agrees_with_restore_on_linked_paths(tmp_path, bad_parent):
    path = tmp_path / "paths.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        names = ["bin/activate", "bin/conda-unpack"]
        names += ["bin", "bin/python"] if bad_parent else ["lib/python3.12"]
        for name in names:
            member = tarfile.TarInfo(name)
            data = b"synthetic file, never execute"
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))
        if not bad_parent:
            for name, target in (
                ("bin/python", "../alias/python3.12"),
                ("alias", "lib"),
            ):
                link = tarfile.TarInfo(name)
                link.type, link.linkname = tarfile.SYMTYPE, target
                archive.addfile(link)
    if bad_parent:
        with pytest.raises(ValueError, match="non-directory parent"):
            be._check_pack(path)
        with pytest.raises(ValueError, match="non-directory parent"):
            be.env_source._safe_extract(path, tmp_path / "restored")
    else:
        be._check_pack(path)
        be.env_source._safe_extract(path, tmp_path / "restored")
        assert (tmp_path / "restored/bin/python").read_bytes().startswith(b"synthetic")


@pytest.mark.parametrize("flash", [True, False])
def test_probe_emits_measured_runtime_facts(monkeypatch, capsys, flash):
    import builtins

    fake_torch = SimpleNamespace(
        __file__=sys.prefix + "/synthetic/torch.py",
        __version__="2.8.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_capability=lambda i: (8, 9),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "flash_attn", SimpleNamespace(__version__="2.8.3.post2")
    )
    monkeypatch.setitem(sys.modules, "causal_conv1d", SimpleNamespace())
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    monkeypatch.setitem(
        sys.modules,
        "typing_extensions",
        SimpleNamespace(__file__=sys.prefix + "/synthetic/typing_extensions.py"),
    )
    monkeypatch.setattr(
        sys,
        "flags",
        SimpleNamespace(
            **{
                **{
                    name: getattr(sys.flags, name)
                    for name in dir(sys.flags)
                    if not name.startswith("_")
                },
                "isolated": 1,
                "no_user_site": 1,
            }
        ),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "1.7.0")
    original_import = builtins.__import__

    def noisy_import(name, *args, **kwargs):
        if name == "torch":
            print("synthetic dependency import diagnostic")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", noisy_import)
    exec(be._verify_code(flash), {})
    captured = capsys.readouterr()
    assert json.loads(captured.out) == runtime_facts(flash)
    assert "synthetic dependency import diagnostic" in captured.err
