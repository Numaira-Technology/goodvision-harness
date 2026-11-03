"""Release integrity and independent updates without cloud credentials."""

import hashlib
import io
import json
import time

import pytest

from tools import r2_delivery as r2


def entry(name, data=b"delivery"):
    return {
        "file": name,
        "key": "objects/" + name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "url": "https://example.invalid/temporary-url",
    }


@pytest.mark.parametrize(
    "names",
    [
        ["../outside"],
        ["/outside"],
        ["a\\b"],
        ["a", "a/b"],
        ["a", "a"],
        ["a", "a.r2-partial"],
    ],
)
def test_invalid_release_is_rejected_before_any_download(tmp_path, names):
    destination = tmp_path / "delivery"
    with pytest.raises(ValueError):
        r2.fetch({"schema": 1, "files": [entry(n) for n in names]}, destination)
    assert not destination.exists()


def test_fetch_resumes_then_verifies_and_preserves_existing_files(
    tmp_path, monkeypatch
):
    data = b"original delivery bytes"
    (tmp_path / "main.zip.r2-partial").write_bytes(data[:8])
    calls = []

    class Response(io.BytesIO):
        status = 206

    class Opener:
        def open(self, request, timeout):
            calls.append(request.get_header("Range"))
            return Response(data[8:])

    monkeypatch.setattr(r2, "build_opener", lambda *_: Opener())
    plan = {
        "schema": 1,
        "expires_at": time.time() + 60,
        "files": [entry("main.zip", data)],
    }
    r2.fetch(plan, tmp_path)
    assert (tmp_path / "main.zip").read_bytes() == data
    assert calls == ["bytes=8-"]
    r2.fetch(plan, tmp_path)
    assert calls == ["bytes=8-"]
    (tmp_path / "main.zip").write_bytes(b"owner data")
    with pytest.raises(ValueError, match="Existing delivery file differs"):
        r2.fetch(plan, tmp_path)
    assert (tmp_path / "main.zip").read_bytes() == b"owner data"


def test_corrupt_download_never_becomes_an_installable_file(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200

    class Opener:
        def open(self, *_args, **_kwargs):
            return Response(b"corrupt!")

    monkeypatch.setattr(r2, "build_opener", lambda *_: Opener())
    with pytest.raises(ValueError, match="checksum differs"):
        r2.fetch(
            {
                "schema": 1,
                "expires_at": time.time() + 60,
                "files": [entry("main.zip", b"delivery")],
            },
            tmp_path,
        )
    assert not (tmp_path / "main.zip").exists()
    assert not (tmp_path / "main.zip.r2-partial").exists()


def test_multipart_resume_reuses_only_matching_parts(tmp_path, monkeypatch):
    part_size = 2 * 1024**2
    data = b"a" * part_size + b"b" * part_size + b"c" * part_size
    path = tmp_path / "main.zip"
    path.write_bytes(data)
    item = entry(path.name, data)
    checkpoint = tmp_path / "resume.json"
    state = {
        item["key"]: {
            "id": "owned-upload",
            "bytes": len(data),
            "sha256": item["sha256"],
            "part_size": part_size,
        }
    }
    uploaded = []

    class S3:
        finished = False

        def get_paginator(self, operation):
            assert operation == "list_parts"
            return self

        def paginate(self, **kwargs):
            assert kwargs["UploadId"] == "owned-upload"
            return [
                {
                    "Parts": [
                        {
                            "PartNumber": 1,
                            "Size": part_size,
                            "ETag": hashlib.md5(data[:part_size]).hexdigest(),
                        },
                        {"PartNumber": 2, "Size": part_size, "ETag": "wrong"},
                    ]
                }
            ]

        def upload_part(self, **kwargs):
            uploaded.append(kwargs["PartNumber"])
            return {"ETag": hashlib.md5(kwargs["Body"]).hexdigest()}

        def complete_multipart_upload(self, **kwargs):
            assert [p["PartNumber"] for p in kwargs["MultipartUpload"]["Parts"]] == [
                1,
                2,
                3,
            ]
            self.finished = True

    s3 = S3()
    monkeypatch.setattr(
        r2,
        "head",
        lambda *_: (
            {
                "ContentLength": len(data),
                "Metadata": {"sha256": item["sha256"]},
            }
            if s3.finished
            else None
        ),
    )
    r2.upload_file(s3, "bucket", tmp_path, item, state, checkpoint, 2)
    assert sorted(uploaded) == [2, 3]
    assert json.loads(checkpoint.read_text()) == {}


def test_new_main_release_reuses_the_existing_model_object(tmp_path, monkeypatch):
    model = entry("models.tar.gz")
    main = entry("main-v2.1.zip", b"new main")
    previous = {"schema": 1, "files": [model, entry("main-v2.0.zip", b"old main")]}
    value = {
        "schema": 1,
        "delivery_version": "v2.0",
        "module_versions": {"main": "v2.1", "bbox": "v2.0"},
        "files": [
            {k: v for k, v in e.items() if k not in ("key", "url")}
            for e in (model, main)
        ],
    }
    monkeypatch.setattr(r2, "inventory", lambda *_: value)
    monkeypatch.setattr(
        r2,
        "head",
        lambda _s, _b, key: (
            {
                "ContentLength": model["bytes"],
            }
            if key == model["key"]
            else None
        ),
    )
    uploaded = []
    monkeypatch.setattr(
        r2, "upload_file", lambda _s, _b, _r, e, *_: uploaded.append(e["file"])
    )

    class S3:
        def put_object(self, **kwargs):
            published = json.loads(kwargs["Body"])
            assert published["files"][0]["key"] == model["key"]
            assert published["module_versions"] == {"main": "v2.1", "bbox": "v2.0"}

    r2.publish(
        S3(), "bucket", tmp_path, "release-2", tmp_path / "resume.json", previous
    )
    assert uploaded == ["main-v2.1.zip"]


def test_missing_object_rejected_before_issuing_a_server_download_plan(monkeypatch):
    class S3:
        def get_object(self, **_kwargs):
            return {
                "Body": io.BytesIO(
                    json.dumps(
                        {
                            "schema": 1,
                            "files": [entry("missing-model.tar.gz")],
                        }
                    ).encode()
                )
            }

        def generate_presigned_url(self, *_args, **_kwargs):
            pytest.fail("A missing model must fail before URL generation")

    monkeypatch.setattr(r2, "head", lambda *_: None)
    with pytest.raises(ValueError, match="Release object is missing"):
        r2.download_plan(S3(), "bucket", "release.json")


def test_existing_release_accepts_equivalent_json_without_reupload(
    tmp_path, monkeypatch
):
    item = entry("README-服务器安装.txt")
    value = {"schema": 1, "files": [item]}
    source = {"schema": 1, "files": [{k: v for k, v in item.items() if k != "key"}]}
    monkeypatch.setattr(r2, "inventory", lambda *_: source)
    monkeypatch.setattr(r2, "head", lambda *_: {"ContentLength": item["bytes"]})

    class S3:
        def get_object(self, **_kwargs):
            return {"Body": io.BytesIO(json.dumps(value, sort_keys=True).encode())}

        def put_object(self, **_kwargs):
            pytest.fail("An equivalent existing release must not be overwritten")

    assert r2.publish(S3(), "bucket", tmp_path, "same", tmp_path / "resume", value) == (
        "releases/same/r2-release-manifest.json"
    )
