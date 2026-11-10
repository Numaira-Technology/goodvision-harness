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
    monkeypatch.delenv("GOOGLE_DRIVE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.mark.parametrize("supplied", [[], ["--samples"], ["--manifest"]])
def test_cli_requires_explicit_samples_and_manifest(
    tmp_path, monkeypatch, capsys, supplied,
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
    monkeypatch.setattr(sys, "argv", [
        "download_media.py", "--samples", str(samples_path),
        "--manifest", str(manifest_path), "--output", str(media), "--check-only",
    ])
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
