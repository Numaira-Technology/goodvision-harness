"""Offline CLI contracts using synthetic bytes, never private benchmark media."""

import json
import sys

import pytest

from tools.event_benchmark import download_media
from tools.event_benchmark import run_eval as runner


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("offline input checks must not download or call a model")

    monkeypatch.setattr(download_media.urllib.request, "urlopen", unexpected_request)
    monkeypatch.setattr(download_media, "open_download", unexpected_request)
    monkeypatch.setattr(download_media.subprocess, "run", unexpected_request)
    monkeypatch.delenv("GOOGLE_DRIVE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.mark.parametrize("supplied", [[], ["--samples"], ["--manifest"]])
def test_cli_requires_explicit_samples_and_manifest(
    tmp_path,
    monkeypatch,
    capsys,
    supplied,
):
    args = ["download_media.py", "--output", str(tmp_path / "media")]
    for flag in supplied:
        args.extend([flag, str(tmp_path / "unused.json")])
    monkeypatch.setattr(sys, "argv", args)

    with pytest.raises(SystemExit) as error:
        download_media.main()

    assert error.value.code == 2
    error_text = capsys.readouterr().err
    assert "the following arguments are required:" in error_text
    missing = error_text.split("the following arguments are required:", 1)[1]
    for flag in {"--samples", "--manifest"} - set(supplied):
        assert flag in missing
    assert not (tmp_path / "media").exists()


@pytest.fixture
def input_pair(tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    clip = media / "unit-test.mp4"
    clip.write_bytes(b"synthetic hash fixture; not a playable benchmark video")
    samples = [
        {
            "number": number,
            "filename": clip.name,
            "event": event,
            "question": f"{event}? Answer yes or no.",
            "proposed_answer": "no",
        }
        for number, event in enumerate(["fight", "fall"], start=1)
    ]
    samples_path = tmp_path / "selected-samples.json"
    samples_path.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    manifest = {
        "status": "human_reviewed",
        "samples_json_sha256": runner.sha256(samples_path),
        "files": [{**sample, "sha256": runner.sha256(clip)} for sample in samples],
    }
    manifest_path = tmp_path / "selected-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_media.py",
            "--samples",
            str(samples_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(media),
            "--check-only",
        ],
    )
    return samples_path, manifest_path, clip


def test_check_only_counts_questions_separately_from_media(input_pair, capsys):
    assert download_media.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "questions": 2,
        "videos": 1,
        "categories": {"fight": 1, "fall": 1},
        "model_requests": 0,
    }


@pytest.mark.parametrize("change", ["samples", "media", "draft"])
def test_check_only_rejects_unfrozen_inputs(input_pair, capsys, change):
    samples, manifest, clip = input_pair
    if change == "samples":
        samples.write_text(samples.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif change == "media":
        clip.write_bytes(b"changed synthetic bytes")
    else:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["status"] = "draft_pending_user_review"
        manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        download_media.main()

    assert capsys.readouterr().out == ""


@pytest.fixture
def transfers(monkeypatch):
    """Intercept every transfer boundary, including credential-free rclone."""
    from email.message import Message
    from io import BytesIO

    calls = []

    def http(body=b"synthetic video", length=None, content_type="video/mp4"):
        def response(request):
            calls.append(request)
            stream = BytesIO(body)
            stream.headers = Message()
            stream.headers["Content-Type"] = content_type
            if length is not None:
                stream.headers["Content-Length"] = str(length)
            return stream

        monkeypatch.setattr(download_media, "open_download", response)

    def rclone(body=b"synthetic video", failure=None):
        def copy(command, **kwargs):
            from pathlib import Path

            calls.append((command, kwargs))
            if body is not None:
                Path(command[5]).write_bytes(body)
            if failure:
                raise failure

        monkeypatch.setattr(download_media.subprocess, "run", copy)

    return http, rclone, calls


@pytest.mark.parametrize(
    "field, reference",
    [
        ("drive_file_id", "unit_test-id"),
        ("url", "unit_test-id"),
        ("video_url", "https://drive.google.com/file/d/unit_test-id/view?usp=sharing"),
        ("url", "https://drive.google.com/open?id=unit_test-id"),
        ("url", "https://drive.google.com/uc?export=download&id=unit_test-id"),
    ],
)
def test_rclone_drive_references_override_environment_token(
    tmp_path,
    transfers,
    field,
    reference,
):
    _, rclone, calls = transfers
    rclone()
    sample = {"filename": "clip.mp4", field: reference}
    download_media.download_video(sample, tmp_path, {}, "unused-secret", "team:")
    command, kwargs = calls[0]
    assert command[:5] == ["rclone", "backend", "copyid", "team:", "unit_test-id"]
    assert "unused-secret" not in str(calls)
    assert kwargs == {"check": True, "capture_output": True}
    assert (tmp_path / "clip.mp4").read_bytes() == b"synthetic video"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.mp4"]


@pytest.mark.parametrize("use_link", [False, True])
def test_http_auth_and_link_precedence(tmp_path, transfers, use_link):
    http, _, calls = transfers
    http(length=15)
    links = {"clip.mp4": "https://download.example/clip"} if use_link else {}
    download_media.download_video(
        {"filename": "clip.mp4", "drive_file_id": "unit-id", "size_bytes": 15},
        tmp_path,
        links,
        "synthetic-token",
        "team" if use_link else None,
    )
    request = calls[0]
    if use_link:
        assert request.full_url == links["clip.mp4"]
        assert request.get_header("Authorization") is None
    else:
        assert request.full_url == (
            "https://www.googleapis.com/drive/v3/files/unit-id"
            "?alt=media&supportsAllDrives=true"
        )
        assert request.get_header("Authorization") == "Bearer synthetic-token"


@pytest.mark.parametrize("method", ["http", "rclone"])
@pytest.mark.parametrize("body, size", [(b"", None), (b"short", 10)])
def test_failed_transfer_preserves_destination_and_cleans_staging(
    tmp_path,
    transfers,
    method,
    body,
    size,
):
    http, rclone, _ = transfers
    destination = tmp_path / "clip.mp4"
    destination.write_bytes(b"")
    if method == "http":
        http(body)
    else:
        rclone(body)
    with pytest.raises(ValueError, match=r"empty file|size differs"):
        download_media.download_video(
            {"filename": "clip.mp4", "drive_file_id": "unit-id", "size_bytes": size},
            tmp_path,
            {},
            "synthetic-token",
            "team" if method == "rclone" else None,
        )
    assert destination.read_bytes() == b""
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(
    "options, error",
    [
        ({"length": 99}, "incomplete HTTP"),
        ({"length": "bad-length"}, "invalid literal"),
        ({"content_type": "text/html"}, "page or error"),
        ({"content_type": "application/json"}, "page or error"),
    ],
)
def test_http_rejects_incomplete_and_error_bodies(tmp_path, transfers, options, error):
    http, _, _ = transfers
    http(**options)
    with pytest.raises(ValueError, match=error):
        download_media.download_video(
            {"filename": "clip.mp4"},
            tmp_path,
            {"clip.mp4": "https://download.example/clip"},
            "",
        )
    assert list(tmp_path.iterdir()) == []


def test_stale_partial_is_not_promoted_after_noop_rclone(tmp_path, transfers):
    _, rclone, _ = transfers
    stale = tmp_path / "clip.mp4.part"
    stale.write_bytes(b"stale incomplete data")
    rclone(body=None)
    with pytest.raises(OSError):
        download_media.download_video(
            {"filename": "clip.mp4", "drive_file_id": "unit-id"},
            tmp_path,
            {},
            "",
            "team",
        )
    assert list(tmp_path.iterdir()) == [stale]
    assert stale.read_bytes() == b"stale incomplete data"


def symlink_or_skip(link, target):
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("Creating symlinks is unavailable on this host")


def test_fixed_partial_symlink_is_never_followed(tmp_path, transfers):
    http, _, _ = transfers
    http()
    target = tmp_path / "unrelated.txt"
    target.write_bytes(b"preserve me")
    symlink_or_skip(tmp_path / "clip.mp4.part", target)
    download_media.download_video(
        {"filename": "clip.mp4"},
        tmp_path,
        {"clip.mp4": "https://download.example/clip"},
        "",
    )
    assert target.read_bytes() == b"preserve me"
    assert (tmp_path / "clip.mp4.part").is_symlink()
    assert (tmp_path / "clip.mp4").read_bytes() == b"synthetic video"


def test_destination_symlink_is_rejected(tmp_path):
    target = tmp_path / "unrelated.txt"
    target.write_bytes(b"preserve me")
    symlink_or_skip(tmp_path / "clip.mp4", target)
    with pytest.raises(ValueError, match="symbolic link"):
        download_media.download_video({"filename": "clip.mp4"}, tmp_path, {}, "")
    assert target.read_bytes() == b"preserve me"


@pytest.mark.parametrize("size", [None, 15])
def test_completed_file_is_idempotent(tmp_path, size):
    destination = tmp_path / "clip.mp4"
    destination.write_bytes(b"synthetic video")
    download_media.download_video(
        {"filename": "clip.mp4", "size_bytes": size},
        tmp_path,
        {},
        "",
    )
    assert destination.read_bytes() == b"synthetic video"


@pytest.mark.parametrize(
    "change",
    [
        {"filename": "../clip.mp4"},
        {"filename": ".."},
        {"filename": ""},
        {"filename": "a\\clip.mp4"},
        {"filename": "C:clip.mp4"},
        {"filename": "a\n.mp4"},
        {"filename": None},
        {"size_bytes": True},
        {"size_bytes": 0},
        {"size_bytes": -1},
        {"size_bytes": "15"},
        {"size_bytes": 15.0},
        {"drive_file_id": "--version"},
        {"drive_file_id": "x/../../y"},
        {"drive_file_id": []},
        {"video_url": 123},
    ],
)
def test_malformed_metadata_is_rejected(tmp_path, change):
    with pytest.raises(ValueError):
        download_media.download_video(
            {"filename": "clip.mp4", **change}, tmp_path, {}, ""
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "remote", ["", "--config=x", ":drive", "team:path", "team::", "x/y"]
)
def test_only_existing_remote_names_are_accepted(remote):
    with pytest.raises(ValueError, match="existing rclone remote"):
        download_media.drive_remote_name(remote)


def test_duplicate_questions_merge_matching_download_metadata():
    rows = [
        {"filename": "bag.mp4", "event": "bag_knife", "drive_file_id": "bag-id"},
        {
            "filename": "bag.mp4",
            "event": "bag_alcohol_bottle",
            "size_bytes": 15,
            "url": "https://drive.google.com/file/d/bag-id/view",
        },
    ]
    assert download_media.unique_videos(rows) == {
        "bag.mp4": {"filename": "bag.mp4", "drive_file_id": "bag-id", "size_bytes": 15},
    }
    assert "size_bytes" not in rows[0]


@pytest.mark.parametrize("change", [{"drive_file_id": "other-id"}, {"size_bytes": 16}])
def test_conflicting_duplicate_metadata_is_rejected(change):
    row = {"filename": "clip.mp4", "drive_file_id": "unit-id", "size_bytes": 15}
    with pytest.raises(ValueError, match="conflicting download metadata"):
        download_media.unique_videos([row, {**row, **change}])


@pytest.mark.parametrize(
    "newurl, authorized",
    [
        ("https://www.googleapis.com/other", True),
        ("https://download.example/clip", False),
        ("https://www.googleapis.com:8443/other", False),
    ],
)
def test_redirects_do_not_leak_drive_token(newurl, authorized):
    request = download_media.urllib.request.Request(
        "https://www.googleapis.com/drive/v3/files/unit-id?alt=media",
        headers={"Authorization": "Bearer synthetic-token"},
    )
    result = download_media.HTTPSRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        newurl,
    )
    assert result.has_header("Authorization") is authorized


@pytest.mark.parametrize("url", ["http://download.example/clip", "file:///tmp/clip"])
def test_redirects_must_remain_https(url):
    request = download_media.urllib.request.Request("https://download.example/first")
    with pytest.raises(ValueError, match="HTTPS"):
        download_media.HTTPSRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            url,
        )


def test_videos_cli_downloads_every_category_and_continues_after_failure(
    tmp_path,
    monkeypatch,
    capsys,
    transfers,
):
    import http.client

    _, rclone, calls = transfers
    rclone()
    path = tmp_path / "videos.json"
    rows = [
        {"filename": "base.mp4", "drive_file_id": "base-id", "event": "fight"},
        {"filename": "bag.mp4", "drive_file_id": "bag-id", "event": "bag_knife"},
        {
            "filename": "bag.mp4",
            "drive_file_id": "bag-id",
            "event": "bag_alcohol_bottle",
        },
    ]
    path.write_text(json.dumps({"videos": rows}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_media.py",
            "--videos",
            str(path),
            "--output",
            str(tmp_path / "media"),
            "--drive-remote",
            "team",
        ],
    )
    original = download_media.download_video

    def partly_broken(sample, *args):
        if sample["filename"] == "base.mp4":
            raise http.client.IncompleteRead(b"partial")
        return original(sample, *args)

    monkeypatch.setattr(download_media, "download_video", partly_broken)
    assert download_media.main() == 1
    assert len(calls) == 1
    captured = capsys.readouterr()
    assert "1/2 videos ready; 1 failed" in captured.out
    assert "base.mp4: IncompleteRead" in captured.err
    assert (tmp_path / "media" / "bag.mp4").read_bytes() == b"synthetic video"


def test_videos_cannot_bypass_review_with_check_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_media.py",
            "--videos",
            str(tmp_path / "unread.json"),
            "--output",
            str(tmp_path),
            "--check-only",
        ],
    )
    with pytest.raises(SystemExit) as error:
        download_media.main()
    assert error.value.code == 2


@pytest.mark.parametrize("link", [None, "", 0, "https://:secret@download.example/clip"])
def test_invalid_explicit_link_does_not_fall_back_to_credentials(tmp_path, link):
    with pytest.raises(ValueError, match="HTTPS"):
        download_media.download_video(
            {"filename": "clip.mp4", "drive_file_id": "unit-id"},
            tmp_path,
            {"clip.mp4": link},
            "synthetic-token",
            "team",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("rows", [[], {}, "bad-list", [None], [{}]])
def test_invalid_catalog_is_rejected_before_creating_output(
    tmp_path, monkeypatch, rows
):
    path = tmp_path / "videos.json"
    path.write_text(json.dumps({"videos": rows}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_media.py",
            "--videos",
            str(path),
            "--output",
            str(tmp_path / "media"),
            "--drive-remote",
            "team",
        ],
    )
    with pytest.raises(SystemExit) as error:
        download_media.main()
    assert error.value.code == 2
    assert not (tmp_path / "media").exists()


def test_interrupted_rclone_cleans_partial_and_preserves_previous_file(
    tmp_path, transfers
):
    import subprocess

    _, rclone, _ = transfers
    destination = tmp_path / "clip.mp4"
    destination.write_bytes(b"old")
    rclone(b"partial", subprocess.CalledProcessError(1, ["rclone"]))
    with pytest.raises(subprocess.CalledProcessError):
        download_media.download_video(
            {"filename": "clip.mp4", "drive_file_id": "unit-id", "size_bytes": 15},
            tmp_path,
            {},
            "",
            "team",
        )
    assert destination.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [destination]


def test_legacy_samples_transfer_replaces_wrong_size_and_preserves_inputs(
    tmp_path,
    monkeypatch,
    transfers,
):
    _, rclone, calls = transfers
    rclone()
    path = tmp_path / "samples.json"
    original = json.dumps(
        {
            "samples": [
                {"filename": "clip.mp4", "drive_file_id": "unit-id", "size_bytes": 15},
            ]
        }
    )
    path.write_text(original, encoding="utf-8")
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.mp4").write_bytes(b"short")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_media.py",
            "--samples",
            str(path),
            "--manifest",
            str(tmp_path / "not-read-during-transfer.json"),
            "--output",
            str(media),
            "--drive-remote",
            "team",
        ],
    )
    assert download_media.main() == 0
    assert download_media.main() == 0
    assert len(calls) == 1
    assert (media / "clip.mp4").read_bytes() == b"synthetic video"
    assert path.read_text(encoding="utf-8") == original


def test_rclone_destination_is_absolute_even_with_older_relative_tempdir(
    tmp_path,
    monkeypatch,
    transfers,
):
    import os
    from contextlib import contextmanager
    from pathlib import Path

    _, rclone, calls = transfers
    rclone()
    monkeypatch.chdir(tmp_path)
    output = Path("team:downloads" if os.name != "nt" else "relative-media")
    output.mkdir()
    real_tempdir = download_media.tempfile.TemporaryDirectory

    @contextmanager
    def relative_tempdir(**kwargs):
        # Python 3.10/3.11 preserve relative dir= paths, unlike Python 3.12+.
        with real_tempdir(**kwargs) as staging:
            yield os.path.relpath(staging)

    monkeypatch.setattr(download_media.tempfile, "TemporaryDirectory", relative_tempdir)
    download_media.download_video(
        {"filename": "clip.mp4", "drive_file_id": "unit-id"},
        output,
        {},
        "",
        "team",
    )
    command, _ = calls[0]
    assert Path(command[5]).is_absolute()
    assert "--ignore-checksum" not in command
    assert "--ignore-times" not in command
    assert (output / "clip.mp4").read_bytes() == b"synthetic video"
