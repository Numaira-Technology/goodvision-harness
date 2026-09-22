"""Successive source updates have different IDs on every destination machine."""

import copy
import hashlib
import io
import json
import tarfile
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from test_harness_code_update import BASE_ID, MANIFEST, FakeEngine

from tools.harness_update.apply_update import upgrade
from tools.harness_update.release_metadata import read_metadata

SOURCE = b"published application source\n"
RELEASES = {
    f"20260921-cloud-health{i}": {"service.py": hashlib.sha256(SOURCE).hexdigest()}
    for i in (4, 5, 6)
}


class ChainEngine(FakeEngine):
    def __init__(self, release="20260921-cloud-health4", image_id="machine-a-health4"):
        super().__init__()
        self.image_id = image_id
        self.release = release
        self.layers = ["original-layer", "source-layer"]
        self.bad_path = None
        self.archive_calls = []
        self.next_image_id = "new-image"
        for target in self.containers.values():
            target["Image"] = image_id
            # Old container release label must not decide compatibility.
            target["Config"]["Labels"]["goodvision.ui.release"] = "stale-label"

    def request(self, method, path, body=None, raw=False):
        decoded = unquote(urlsplit(path).path)
        if method == "GET" and decoded == f"/images/{self.image_id}/json":
            return {
                "Id": self.image_id,
                "RootFS": {"Layers": self.layers},
                "Config": {"Labels": {"goodvision.ui.release": self.release}},
            }
        if method == "GET" and decoded == f"/images/{BASE_ID}/json":
            return {"Id": BASE_ID, "RootFS": {"Layers": ["original-layer"]}}
        if decoded.endswith("/archive"):
            source_path = parse_qs(urlsplit(path).query)["path"][0]
            self.archive_calls.append((decoded, source_path))
            content = (
                b"changed" if self.bad_path and self.bad_path in source_path else SOURCE
            )
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w") as archive:
                member = tarfile.TarInfo("service.py")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
            return stream.getvalue()
        if method == "GET" and decoded == "/images/updated/json":
            return {"Id": self.next_image_id}
        result = super().request(method, path, body)
        if decoded == "/containers/create":
            self.containers[result["Id"]]["Image"] = self.next_image_id
        return result


@pytest.mark.parametrize("release", RELEASES)
@pytest.mark.parametrize(
    "image_id", ["machine-a-custom-build-id", "machine-b-different-build-id"]
)
@pytest.mark.parametrize("check_only", [True, False])
def test_published_updates_accept_machine_specific_ids(
    tmp_path, release, image_id, check_only
):
    engine = ChainEngine(release, image_id)
    upgrade(
        engine,
        MANIFEST | {"published_sources": RELEASES},
        "updated",
        tmp_path / "state.json",
        check_only=check_only,
        base_id=image_id,
        verification_base_id=BASE_ID,
    )
    assert len(engine.archive_calls) == 4  # Both application paths on both lanes.
    assert len(engine.created) == (0 if check_only else 2)
    if check_only:
        assert all(method == "GET" for method, _, _ in engine.calls)


@pytest.mark.parametrize(
    "failure",
    [
        "unknown-release",
        "wrong-ancestry",
        "untrusted-helper",
        "site-source",
        "app-source",
        "mixed-lanes",
    ],
)
def test_invalid_update_chains_stop_before_container_changes(tmp_path, failure):
    engine = ChainEngine()
    helper = BASE_ID
    if failure == "unknown-release":
        engine.release = "unpublished"
    elif failure == "wrong-ancestry":
        engine.layers = ["unrelated-layer", "source-layer"]
    elif failure == "untrusted-helper":
        helper = "unverified"
    elif failure == "site-source":
        engine.bad_path = "site-packages"
    elif failure == "app-source":
        engine.bad_path = "/app/src"
    elif failure == "mixed-lanes":
        engine.containers["old2"]["Image"] = "another-build"
    before = copy.deepcopy(engine.containers)
    with pytest.raises(RuntimeError):
        upgrade(
            engine,
            MANIFEST | {"published_sources": RELEASES},
            "updated",
            tmp_path / "state.json",
            base_id=engine.image_id,
            verification_base_id=helper,
        )
    assert engine.containers == before
    assert all(method == "GET" for method, _, _ in engine.calls)
    assert not (tmp_path / "state.json").exists()


def test_successive_releases_keep_windows_launcher_valid(tmp_path):
    installed = tmp_path / "installation"
    installed.mkdir()
    files = {
        "RELEASE.json": json.dumps(
            {"accepted_image_ids": [BASE_ID], "image": "original.tar"}
        ).encode(),
        "SOURCE_HASHES.json": json.dumps({"image_id": BASE_ID, "files": {}}).encode(),
    }
    files["FILES.json"] = json.dumps(
        {
            "files": [
                {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()
            ]
        }
    ).encode()
    for name, data in files.items():
        (installed / name).write_bytes(data)
    previous_id, previous_release = BASE_ID, "initial"
    for version in (4, 5, 6):
        release = f"20260921-cloud-health{version}"
        engine = ChainEngine(previous_release, previous_id)
        engine.next_image_id = f"machine-specific-built-image-{version}"
        manifest = MANIFEST | {
            "release": release,
            "published_sources": RELEASES,
            "files": [
                {
                    "path": "payload/service.py",
                    "sha256": hashlib.sha256(SOURCE).hexdigest(),
                }
            ],
        }
        upgrade(
            engine,
            manifest,
            "updated",
            tmp_path / f"state-{version}.json",
            base_id=previous_id,
            verification_base_id=BASE_ID,
            installation=installed,
        )
        assert len(engine.created) == 2
        read_metadata(installed, engine.next_image_id)
        metadata = json.loads((installed / "RELEASE.json").read_text())
        assert metadata["ui_update"] == {
            "release": release,
            "image_id": engine.next_image_id,
        }
        assert metadata["image"] == "original.tar"
        previous_id, previous_release = engine.next_image_id, release
