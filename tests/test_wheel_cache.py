"""Prebuilt CUDA extension wheels: pins, fetch, create-only upload, builder."""

import copy
import hashlib
import io
import json
import traceback
from contextlib import contextmanager
from pathlib import Path

import pytest
from wheel_fixtures import wheel_bytes

from tools import build_env as be
from tools import registry, wheel_cache

COMPAT = {"python": "3.12", "cuda": "12.8", "torch": "2.8.0", "gpu_family": "ada"}
FLASH = "flash_attn-2.8.3.post2-cp312-cp312-linux_x86_64.whl"
CONV = "causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl"


def pin_for(name, version, filename, data, compat=COMPAT):
    sha = hashlib.sha256(data).hexdigest()
    return {
        "name": name,
        "version": version,
        "filename": filename,
        "compat": dict(compat),
        "r2_key": f"wheels/{name}/{sha}.whl",
        "sha256": sha,
        "size": len(data),
    }


class FakeR2:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts = []
        self.gets = []

    def get_object(self, Bucket, Key):
        self.gets.append(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, ContentLength=None):
        assert IfNoneMatch == "*"
        data = Body.read() if hasattr(Body, "read") else Body
        self.puts.append(Key)
        if IfNoneMatch == "*" and Key in self.objects:
            error = RuntimeError("exists")
            error.response = {"Error": {"Code": "PreconditionFailed"}}
            raise error
        self.objects[Key] = data


def test_pins_must_be_content_addressed_and_complete():
    good = pin_for("flash-attn", "2.8.3.post2", FLASH, b"w")
    assert wheel_cache.validate_pin(good) is good
    for bad in (
        {**good, "r2_key": "wheels/flash-attn/latest.whl"},
        {**good, "sha256": "0" * 63},
        {**good, "filename": "flash_attn.tar.gz"},
        {**good, "compat": {"python": "3.12"}},
        {**good, "size": 0},
    ):
        with pytest.raises(registry.RegistryError):
            wheel_cache.validate_pin(bad)


def test_pins_for_matches_exact_compat_and_compiled_versions(tmp_path):
    flash = pin_for("flash-attn", "2.8.3.post2", FLASH, b"a")
    hopper = pin_for(
        "causal-conv1d", "1.7.0", CONV, b"b", {**COMPAT, "gpu_family": "hopper"}
    )
    old = pin_for("causal-conv1d", "1.6.0", CONV.replace("1.7.0", "1.6.0"), b"c")
    import yaml

    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump({"version": 1, "models": {}, "wheels": [flash, hopper, old]})
    )
    assert set(wheel_cache.pins_for(COMPAT, path)) == {"flash-attn"}


def test_fetch_verifies_size_and_hash(tmp_path):
    data = wheel_bytes(FLASH)
    pin = pin_for("flash-attn", "2.8.3.post2", FLASH, data)
    got = wheel_cache.fetch(
        pin, tmp_path, client=FakeR2({pin["r2_key"]: data}), bucket="b"
    )
    assert got.name == FLASH and got.read_bytes() == data
    with pytest.raises(RuntimeError, match="pinned wheel fetch failed"):
        wheel_cache.fetch(
            pin,
            tmp_path / "x",
            client=FakeR2({pin["r2_key"]: b"tampered bytes!"}),
            bucket="b",
        )
    assert not list((tmp_path / "x").rglob("*.part"))
    assert not list((tmp_path / "x").rglob("*.whl"))


def test_upload_is_create_only_and_idempotent_for_identical_bytes(tmp_path):
    wheel = tmp_path / FLASH
    data = wheel_bytes(FLASH)
    wheel.write_bytes(data)
    client = FakeR2()
    pin = wheel_cache.upload(
        wheel, "flash-attn", "2.8.3.post2", COMPAT, client=client, bucket="b"
    )
    assert client.objects[pin["r2_key"]] == data
    assert pin["r2_key"][:-4] + ".json" in client.objects
    again = wheel_cache.upload(
        wheel, "flash-attn", "2.8.3.post2", COMPAT, client=client, bucket="b"
    )
    assert again == pin and client.objects[pin["r2_key"]] == data
    assert client.gets == [pin["r2_key"], pin["r2_key"][:-4] + ".json"]
    with pytest.raises(ValueError):
        wheel_cache.upload(wheel, "numpy", "2.0", COMPAT, client=client, bucket="b")


def test_name_version_from_filename():
    assert wheel_cache.name_version(FLASH) == ("flash-attn", "2.8.3.post2")
    assert wheel_cache.name_version(CONV) == ("causal-conv1d", "1.7.0")


def builder(tmp_path, monkeypatch, *, pins, fetch):
    env = copy.deepcopy(registry.env_of("videojev"))
    env["unpack_to"] = str(tmp_path / "prefix")
    out = tmp_path / "new.tar.gz"
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if "wheel" in argv and "-w" in argv:
            d = Path(argv[argv.index("-w") + 1])
            d.mkdir(parents=True, exist_ok=True)
            for name in (FLASH, CONV):
                (d / name).write_bytes(wheel_bytes(name))

    monkeypatch.setattr(
        be, "_write_constraints", lambda prefix, path: path.write_text("")
    )
    monkeypatch.setattr(be.registry, "env_of", lambda _: env)
    monkeypatch.setattr(be.env_source, "check_compatibility", lambda _: None)
    monkeypatch.setattr(be, "_ensure_miniconda", lambda *a: tmp_path / "bin/conda")
    monkeypatch.setattr(be, "_run", run)
    monkeypatch.setattr(be, "_pack", lambda *a: out.write_bytes(b"pack"))
    monkeypatch.setattr(be, "check_conda_ownership", lambda prefix: None)
    monkeypatch.setattr(be, "verify_pack", lambda tarball: {"dists": 0})
    monkeypatch.setattr(be.wheel_cache, "pins_for", lambda compat: pins)
    monkeypatch.setattr(be.wheel_cache, "fetch", fetch)
    return out, calls


def test_builder_installs_pinned_wheels_without_compiling(
    tmp_path, monkeypatch, capsys
):
    pins = {
        "flash-attn": pin_for("flash-attn", "2.8.3.post2", FLASH, wheel_bytes(FLASH)),
        "causal-conv1d": pin_for("causal-conv1d", "1.7.0", CONV, wheel_bytes(CONV)),
    }

    def fetch(pin, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        path = Path(dest) / pin["filename"]
        path.write_bytes(wheel_bytes(pin["filename"]))
        return path

    out, calls = builder(tmp_path, monkeypatch, pins=pins, fetch=fetch)
    be.main(["--model", "videojev", "--out", str(out), "--skip-upload"])
    assert not any("wheel" in argv and "-w" in argv for argv in calls)  # no compile
    installs = [argv for argv in calls if any(str(a).endswith(FLASH) for a in argv)]
    assert installs and "--constraint" in installs[0]
    assert {"--no-deps", "--no-index", "--force-reinstall"} <= set(installs[0])
    assert any(argv[-1] == "check" for argv in calls)
    report = capsys.readouterr().out
    assert '"source": "pinned"' in report and "fetched and verified" in report


def test_builder_compiles_and_keeps_wheels_when_a_fetch_fails(
    tmp_path, monkeypatch, capsys
):
    pins = {
        "flash-attn": pin_for("flash-attn", "2.8.3.post2", FLASH, b"a"),
        "causal-conv1d": pin_for("causal-conv1d", "1.7.0", CONV, b"b"),
    }

    def fetch(pin, dest):
        raise RuntimeError("size or SHA-256 differs")

    out, calls = builder(tmp_path, monkeypatch, pins=pins, fetch=fetch)
    be.main(["--model", "videojev", "--out", str(out), "--skip-upload"])
    assert any("wheel" in argv and "-w" in argv for argv in calls)  # compiled
    kept = sorted(
        p.name for p in (tmp_path / "new.tar.gz.wheels/compiled").glob("*.whl")
    )
    assert kept == sorted([FLASH, CONV])
    report = capsys.readouterr().out
    assert '"source": "compiled"' in report and "fetch failed" in report


def test_builder_compiles_when_no_complete_pin_set(tmp_path, monkeypatch, capsys):
    out, calls = builder(
        tmp_path,
        monkeypatch,
        pins={},
        fetch=lambda *a: pytest.fail("nothing pinned to fetch"),
    )
    be.main(["--model", "videojev", "--out", str(out), "--skip-upload"])
    assert any("wheel" in argv and "-w" in argv for argv in calls)
    compile_call = next(argv for argv in calls if "wheel" in argv and "-w" in argv)
    assert "--no-binary=flash-attn,causal-conv1d" in compile_call
    assert "--constraint" in compile_call and "--no-deps" in compile_call


@pytest.mark.parametrize("change", [
    {"filename": FLASH.replace("cp312", "cp311")},
    {"filename": CONV}, {"version": "2.8.2"},
    {"compat": dict.fromkeys(COMPAT)},
    {"compat": {**COMPAT, "cuda": 12.8}},
    {"compat": {**COMPAT, "gpu_family": "unknown"}},
])
def test_pin_identity_and_compatibility_fail_before_remote_reads(tmp_path, change):
    pin = pin_for("flash-attn", "2.8.3.post2", FLASH, wheel_bytes(FLASH))
    client = FakeR2()
    with pytest.raises(registry.RegistryError):
        wheel_cache.fetch({**pin, **change}, tmp_path, client=client, bucket="b")
    assert not client.gets and not client.puts


@pytest.mark.parametrize("metadata", [
    {"metadata_name": "causal-conv1d"}, {"tag": "cp311-cp311-linux_x86_64"},
])
def test_upload_checks_benign_wheel_metadata_before_remote_writes(tmp_path, metadata):
    path = tmp_path / FLASH
    path.write_bytes(wheel_bytes(FLASH, **metadata))
    client = FakeR2()
    with pytest.raises(ValueError, match=r"metadata|METADATA"):
        wheel_cache.upload(
            path, "flash-attn", "2.8.3.post2", COMPAT, client=client, bucket="b"
        )
    assert not client.puts


def test_shared_cache_keeps_distinct_benign_builds_and_reuses_verified_bytes(tmp_path):
    data_a = wheel_bytes(FLASH, label=b"build a")
    data_b = wheel_bytes(FLASH, label=b"build b")
    pin_a = pin_for("flash-attn", "2.8.3.post2", FLASH, data_a)
    pin_b = pin_for("flash-attn", "2.8.3.post2", FLASH, data_b)
    client = FakeR2({pin_a["r2_key"]: data_a, pin_b["r2_key"]: data_b})
    a = wheel_cache.fetch(pin_a, tmp_path, client=client, bucket="b")
    inode = a.stat().st_ino
    b = wheel_cache.fetch(pin_b, tmp_path, client=client, bucket="b")
    assert a != b and a.read_bytes() == data_a and b.read_bytes() == data_b
    assert wheel_cache.fetch(pin_a, tmp_path, client=client, bucket="b") == a
    assert a.stat().st_ino == inode and len(client.gets) == 2


@pytest.mark.parametrize("conflict", ["bytes", "sidecar"])
def test_upload_preserves_conflicting_immutable_objects(tmp_path, conflict):
    data = wheel_bytes(FLASH, label=b"build a")
    path = tmp_path / FLASH
    path.write_bytes(data)
    pin = pin_for("flash-attn", "2.8.3.post2", FLASH, data)
    other = wheel_bytes(FLASH, label=b"build b")
    assert len(other) == len(data)  # equal length alone is insufficient
    sidecar_key = pin["r2_key"][:-4] + ".json"
    other_sidecar = {**pin, "compat": {**COMPAT, "torch": "2.9.0"}}
    client = FakeR2({
        pin["r2_key"]: other if conflict == "bytes" else data,
        sidecar_key: json.dumps(other_sidecar).encode(),
    })
    before = dict(client.objects)
    with pytest.raises(RuntimeError, match="failed or is uncertain"):
        wheel_cache.upload(
            path, "flash-attn", "2.8.3.post2", COMPAT, client=client, bucket="b"
        )
    assert client.objects == before
    assert len(client.puts) == (1 if conflict == "bytes" else 2)


@pytest.mark.parametrize("stage", ["put", "get"])
@pytest.mark.parametrize("response", [
    None, {"Error": None}, {"Error": {"Code": 123}},
    {"Error": {"Code": "PRIVATE_DETAIL"}},
])
def test_upload_sanitizes_put_and_conflict_verification_errors(
    tmp_path, stage, response
):
    path = tmp_path / FLASH
    data = wheel_bytes(FLASH)
    path.write_bytes(data)
    pin = pin_for("flash-attn", "2.8.3.post2", FLASH, data)
    client = FakeR2({pin["r2_key"]: data})
    error = type("PRIVATE_CLASS", (Exception,), {})("PRIVATE_MESSAGE")
    error.response = response

    def fail(**kwargs):
        raise error

    setattr(client, stage + "_object", fail)
    with pytest.raises(RuntimeError, match=r"uncertain.*unclassified error") as exc:
        wheel_cache.upload(
            path, "flash-attn", "2.8.3.post2", COMPAT, client=client, bucket="b"
        )
    assert "PRIVATE_" not in "".join(traceback.format_exception(exc.value))
    assert exc.value.__suppress_context__


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_fetch_closes_temporary_file_and_body_before_cleanup(
    tmp_path, monkeypatch, failure
):
    pin = pin_for("flash-attn", "2.8.3.post2", FLASH, wheel_bytes(FLASH))
    handles = []
    real_temp = wheel_cache.tempfile.NamedTemporaryFile
    real_unlink = Path.unlink

    @contextmanager
    def temporary(*args, **kwargs):
        with real_temp(*args, **kwargs) as handle:
            handles.append(handle)
            yield handle

    def unlink(path, *args, **kwargs):
        assert all(handle.closed for handle in handles)
        return real_unlink(path, *args, **kwargs)

    class Body(io.BytesIO):
        def read(self, *args):
            raise failure("read interrupted")

    body = Body()
    client = FakeR2()
    client.get_object = lambda **kwargs: {"Body": body}
    monkeypatch.setattr(wheel_cache.tempfile, "NamedTemporaryFile", temporary)
    monkeypatch.setattr(Path, "unlink", unlink)
    expected = KeyboardInterrupt if failure is KeyboardInterrupt else RuntimeError
    with pytest.raises(expected):
        wheel_cache.fetch(pin, tmp_path, client=client, bucket="b")
    assert body.closed and not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.rglob("*.whl"))


@pytest.mark.parametrize("existing", ["prefix", "compiled"])
def test_builder_preserves_existing_work_before_fetch_or_build(
    tmp_path, monkeypatch, existing
):
    pins = {} if existing == "compiled" else {
        "flash-attn": pin_for("flash-attn", "2.8.3.post2", FLASH, wheel_bytes(FLASH)),
        "causal-conv1d": pin_for("causal-conv1d", "1.7.0", CONV, wheel_bytes(CONV)),
    }
    out, calls = builder(
        tmp_path, monkeypatch, pins=pins,
        fetch=lambda *a: pytest.fail("existing work must be checked first"),
    )
    directory = (tmp_path / "prefix" if existing == "prefix" else
                 tmp_path / "new.tar.gz.wheels/compiled")
    directory.mkdir(parents=True)
    keep = directory / "keep.txt"
    keep.write_text("previous work")
    with pytest.raises(SystemExit):
        be.main(["--model", "videojev", "--out", str(out), "--skip-upload"])
    assert not calls and keep.read_text() == "previous work"


def test_builder_rechecks_pinned_bytes_before_installer(tmp_path, monkeypatch):
    pins = {
        "flash-attn": pin_for("flash-attn", "2.8.3.post2", FLASH, wheel_bytes(FLASH)),
        "causal-conv1d": pin_for("causal-conv1d", "1.7.0", CONV, wheel_bytes(CONV)),
    }

    def changed_file(pin, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        path = Path(dest) / pin["filename"]
        path.write_bytes(wheel_bytes(pin["filename"], label=b"another build"))
        return path

    out, calls = builder(tmp_path, monkeypatch, pins=pins, fetch=changed_file)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        be.main(["--model", "videojev", "--out", str(out), "--skip-upload"])
    assert not any(any(str(arg).endswith(".whl") for arg in argv) for argv in calls)


def test_upload_cli_keeps_producer_declarations_distinct_from_verification(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / FLASH
    path.write_bytes(wheel_bytes(FLASH))
    client = FakeR2()
    monkeypatch.setattr(wheel_cache.model_source, "r2_client", lambda: client)
    monkeypatch.setenv("R2_BUCKET", "synthetic")
    wheel_cache.main(["upload", "--model", "videojev", str(path)])
    pin = json.loads(capsys.readouterr().out)["wheels"][0]
    sidecar = json.loads(client.objects[pin["r2_key"][:-4] + ".json"])
    assert sidecar["uploaded_by"] == "goodvision-harness wheel_cache"
    assert "producer-declared" in sidecar["compatibility_evidence"]
    assert "built_by" not in sidecar


def test_force_compile_does_not_read_wheel_pins(tmp_path, monkeypatch):
    out, calls = builder(
        tmp_path, monkeypatch, pins={},
        fetch=lambda *a: pytest.fail("forced compile must not fetch"),
    )
    monkeypatch.setattr(
        be.wheel_cache, "pins_for",
        lambda *a: pytest.fail("forced compile must not require wheel pins"),
    )
    be.main([
        "--model", "videojev", "--out", str(out), "--skip-upload", "--compile-wheels",
    ])
    assert any("wheel" in argv and "-w" in argv for argv in calls)


def test_existing_cache_mismatch_is_preserved(tmp_path):
    data = wheel_bytes(FLASH)
    pin = pin_for("flash-attn", "2.8.3.post2", FLASH, data)
    final = tmp_path / pin["sha256"] / FLASH
    final.parent.mkdir()
    other = wheel_bytes(FLASH, label=b"different build")
    final.write_bytes(other)
    client = FakeR2({pin["r2_key"]: data})
    with pytest.raises(ValueError, match="SHA-256 differs"):
        wheel_cache.fetch(pin, tmp_path, client=client, bucket="b")
    assert final.read_bytes() == other and not client.gets
