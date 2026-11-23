"""Prebuilt CUDA extension wheels: pins, fetch, create-only upload, builder."""

import copy
import hashlib
import io
from pathlib import Path

import pytest

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

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, ContentLength=None):
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
    data = b"real wheel bytes"
    pin = pin_for("flash-attn", "2.8.3.post2", FLASH, data)
    got = wheel_cache.fetch(
        pin, tmp_path, client=FakeR2({pin["r2_key"]: data}), bucket="b"
    )
    assert got.name == FLASH and got.read_bytes() == data
    with pytest.raises(RuntimeError, match=r"differs|larger"):
        wheel_cache.fetch(
            pin,
            tmp_path / "x",
            client=FakeR2({pin["r2_key"]: b"tampered bytes!"}),
            bucket="b",
        )
    assert not list((tmp_path / "x").glob("*"))  # nothing left behind


def test_upload_is_create_only_and_idempotent_for_identical_bytes(tmp_path):
    wheel = tmp_path / FLASH
    wheel.write_bytes(b"compiled")
    client = FakeR2()
    pin = wheel_cache.upload(
        wheel, "flash-attn", "2.8.3.post2", COMPAT, client=client, bucket="b"
    )
    assert client.objects[pin["r2_key"]] == b"compiled"
    assert pin["r2_key"][:-4] + ".json" in client.objects
    again = wheel_cache.upload(
        wheel, "flash-attn", "2.8.3.post2", COMPAT, client=client, bucket="b"
    )
    assert again == pin and client.objects[pin["r2_key"]] == b"compiled"
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
                (d / name).write_bytes(b"compiled " + name.encode())

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
        "flash-attn": pin_for("flash-attn", "2.8.3.post2", FLASH, b"a"),
        "causal-conv1d": pin_for("causal-conv1d", "1.7.0", CONV, b"b"),
    }

    def fetch(pin, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        path = Path(dest) / pin["filename"]
        path.write_bytes(b"verified")
        return path

    out, calls = builder(tmp_path, monkeypatch, pins=pins, fetch=fetch)
    be.main(["--model", "videojev", "--out", str(out), "--skip-upload"])
    assert not any("wheel" in argv and "-w" in argv for argv in calls)  # no compile
    installs = [argv for argv in calls if any(str(a).endswith(FLASH) for a in argv)]
    assert installs and "--constraint" in installs[0]
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
