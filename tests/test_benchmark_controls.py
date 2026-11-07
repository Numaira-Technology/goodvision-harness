"""Offline routing, precision and split regressions; all model replies are synthetic."""

import io
import json
from pathlib import Path
import re
import shlex

import pytest

from tools.event_benchmark import prepare_provider, run_eval as runner

evidence = runner.evidence


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected network access")
    monkeypatch.setattr(evidence.urllib.request, "urlopen", forbidden)


def provider(tmp_path):
    path = tmp_path / "provider.json"
    evidence.write_new(path, {"schema_version": 1, "model": "model-a", "backend": "decision",
                             "selected_at": "synthetic", "selection": {"method": "synthetic"},
                             "expected_provider_name": "Provider A",
                             "provider": {"only": ["provider-a/fp16"], "allow_fallbacks": False}})
    return path, evidence.load_provider(path, "model-a", "decision")


def reply(score, **fields):
    return {"id": "synthetic-generation", "provider": "Provider A", "model": "model-a",
            "usage": {"input_tokens": 20, "cost": 0.001},
            "answers": {"answer": {"type": "noul", "noul": score}}, **fields}


def split_fixture(tmp_path):
    sidecar = {"schema_version": 1, "status": "validation_membership_and_labels_frozen_with_recorded_limits",
               "train": {"membership": None}, "development": {"membership": None}, "limits": ["not scene-disjoint"]}
    paths = {}
    for part in ("validation", "test"):
        media = tmp_path / part
        media.mkdir()
        clip = media / f"{part}.mp4"
        clip.write_bytes(part.encode())
        row = {"number": 1, "filename": clip.name, "event": "fight", "proposed_answer": "no",
               "human_review": "user_reviewed", "question": "Fight? Answer yes or no."}
        samples, manifest = tmp_path / f"{part}-samples.json", tmp_path / f"{part}-manifest.json"
        evidence.write_new(samples, {"samples": [row]})
        evidence.write_new(manifest, {"status": "human_reviewed", "samples_json_sha256": runner.sha256(samples),
                                      "files": [{**row, "sha256": runner.sha256(clip)}]})
        sidecar[part] = {"questions": 1, "videos": 1, "samples_json_sha256": runner.sha256(samples),
                         "manifest_sha256": runner.sha256(manifest), "members": [
                             {"filename": clip.name, "sha256": runner.sha256(clip),
                              "source_video_group": part + "-source", "question_numbers": [1]}]}
        paths[part] = (samples, manifest, media)
    path = tmp_path / "split.json"
    evidence.write_new(path, sidecar)
    return path, sidecar, paths


def test_lossless_score_does_not_cross_the_fixed_gate(tmp_path, monkeypatch):
    _, context = provider(tmp_path)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"synthetic")
    monkeypatch.setattr(runner, "probe_duration", lambda _: 2.0)
    monkeypatch.setattr(runner, "frame_uri", lambda *args: "synthetic-image")
    monkeypatch.setattr(runner, "_post", lambda *args: reply(0.49996))
    sample = {"filename": clip.name, "question": "Fight? Answer yes or no.", "proposed_answer": "no"}
    _, result = runner.eval_video("decision", "model-a", sample, {clip.name: clip}, "synthetic-key", 1.0, context)
    assert result["score"] == 0.49996 and result["score_display"] == 0.5
    assert result["pred"] == "no" and result["correct"]
    assert [frame["score"] for frame in result["frames"]] == [0.49996, 0.49996]
    assert json.loads(json.dumps(result))["score"] == 0.49996
    assert result["frames"][0]["attempts"][0]["actual_provider"] == "Provider A"


def test_retries_keep_receipts_and_never_change_provider(tmp_path, monkeypatch):
    _, context = provider(tmp_path)
    replies, requests = iter([reply("invalid"), reply(0.123456789123)]), []
    monkeypatch.setattr(runner, "_post", lambda *args: requests.append(args) or next(replies))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    attempts = []
    score = runner.frame_score("decision", "model-a", "Question", "image", "secret-key", context, attempts)
    assert score == 0.123456789123
    assert [a["ok"] for a in attempts] == [False, True]
    assert all(r[1]["provider"] == context["config"]["provider"] for r in requests)
    assert "secret-key" not in json.dumps(attempts)
    assert attempts[0]["actual_provider"] == "Provider A"


@pytest.mark.parametrize("backend", ["language", "decision"])
def test_routing_is_the_only_change_to_request_envelope(tmp_path, backend):
    _, context = provider(tmp_path)
    url, before = runner.build_request(backend, "model-a", "Question", "image")
    new_url, after = runner.build_request(backend, "model-a", "Question", "image", context)
    assert url == new_url
    assert after.pop("provider") == context["config"]["provider"]
    assert before == after


@pytest.mark.parametrize("change", ["fallback", "multiple", "model"])
def test_bad_provider_configuration_rejected_before_inference(tmp_path, change):
    path, context = provider(tmp_path)
    data = context["config"]
    if change == "fallback":
        data["provider"]["allow_fallbacks"] = True
    elif change == "multiple":
        data["provider"]["only"].append("provider-b")
    else:
        data["model"] = "model-b"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        evidence.load_provider(path, "model-a", "decision")


def test_missing_provider_metadata_is_never_inferred_from_requested_provider(tmp_path):
    _, context = provider(tmp_path)
    summary = evidence.provider_summary([{"ok": True, "score": 0.8}], context)
    assert summary["complete"] is False
    assert summary["actual_providers"] == []


def test_generation_metadata_lookup_is_get_only_and_preserves_score(monkeypatch):
    calls = []
    def fetch(request, **kwargs):
        calls.append(request)
        return io.BytesIO(json.dumps({"data": {"id": "synthetic-id", "provider_name": "Provider A",
                                              "model": "model-a", "total_cost": 0.001}}).encode())
    monkeypatch.setattr(evidence.urllib.request, "urlopen", fetch)
    attempts = [{"ok": True, "score": 0.49996, "generation_id": "synthetic-id"}]
    evidence.enrich_attempts(attempts, "secret-key")
    assert len(calls) == 1 and calls[0].get_method() == "GET"
    assert attempts[0]["score"] == 0.49996
    assert attempts[0]["actual_provider"] == "Provider A"
    assert "secret-key" not in json.dumps(attempts)


def test_source_sidecar_join_preserves_original_manifest(tmp_path):
    path, _, paths = split_fixture(tmp_path)
    manifest = paths["test"][1]
    original = manifest.read_bytes()
    groups, context = evidence.split_groups(path, manifest, "test")
    assert groups == {1: "test-source"}
    assert context["train"]["membership"] is None
    assert context["limits"] == ["not scene-disjoint"]
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("change", ["source_overlap", "hash_overlap", "filename_overlap", "identity", "duplicate", "manifest_hash"])
def test_invalid_source_sidecar_is_rejected(tmp_path, change):
    path, data, paths = split_fixture(tmp_path)
    member = data["test"]["members"][0]
    if change.endswith("overlap"):
        field = {"source_overlap": "source_video_group", "hash_overlap": "sha256", "filename_overlap": "filename"}[change]
        member[field] = data["validation"]["members"][0][field]
    elif change == "identity":
        member["sha256"] = "f" * 64
    elif change == "duplicate":
        member["question_numbers"].append(1)
    else:
        data["test"]["manifest_sha256"] = "f" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        evidence.split_groups(path, paths["test"][1], "test")


def test_preflight_never_reads_credentials_or_calls_model(tmp_path, monkeypatch):
    path, _, paths = split_fixture(tmp_path)
    samples, manifest, media = paths["validation"]
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must not read credentials or invoke inference")
    monkeypatch.setattr(runner, "load_key", forbidden)
    monkeypatch.setattr(runner, "_post", forbidden)
    monkeypatch.setattr(runner.sys, "argv", ["run_eval", "--candidate", "luna", "--preflight",
        "--samples", str(samples), "--manifest", str(manifest), "--media", str(media),
        "--split-manifest", str(path), "--split", "validation"])
    result = runner.main()
    assert result["inference_calls"] == 0 and result["data_checks_passed"]
    assert result["split_context"]["split"] == "validation"
    assert "frozen provider configuration" in result["pending"]


def catalog():
    return {"data": {"id": "model-a", "endpoints": [
        {"tag": tag, "provider_name": tag, "status": 0, "supported_parameters": ["temperature", "max_tokens"],
         "latency_last_30m": latency}
        for tag, latency in [("provider-a/fp16", 0.4), ("provider-b/fp16", 0.2)]]}}


def test_fastest_selection_uses_latency_not_price_and_freezes_one_endpoint():
    result = prepare_provider.select_provider(catalog(), "model-a", "language", "a" * 64, "synthetic")
    assert result["provider"]["only"] == ["provider-b/fp16"]
    assert result["provider"]["allow_fallbacks"] is False


def test_missing_latency_does_not_produce_a_fake_fastest_provider():
    data = catalog()
    data["data"]["endpoints"][0]["latency_last_30m"] = None
    with pytest.raises(ValueError, match="fastest provider unresolved"):
        prepare_provider.select_provider(data, "model-a", "language", "a" * 64, "synthetic")


def test_only_endpoint_is_not_labeled_a_measured_fastest_provider():
    data = catalog()
    data["data"]["endpoints"].pop()
    data["data"]["endpoints"][0]["latency_last_30m"] = None
    result = prepare_provider.select_provider(data, "model-a", "language", "a" * 64, "synthetic")
    assert result["selection"]["method"] == "only_eligible_endpoint_not_a_speed_comparison"


def test_robot41_documented_command_runs_legacy_with_synthetic_replies(tmp_path, monkeypatch):
    skill = Path(".claude/skills/event-benchmark/SKILL.md").read_text()
    command = next(block for block in re.findall(r"```sh\n(.*?)```", skill, re.DOTALL)
                   if "run_eval.py" in block and "samples_robot41.json" in block)
    argv = shlex.split(command.replace("\\\n", ""))[1:]
    argv[argv.index("--out") + 1] = str(tmp_path / "robot41.json")
    samples = json.loads(Path(argv[argv.index("--samples") + 1]).read_text())["samples"]
    assert len(samples) == 41
    clip = tmp_path / "synthetic.mp4"
    clip.write_bytes(b"synthetic media; not decoded")
    monkeypatch.setattr(runner, "frozen_inputs", lambda *args:
        (samples, {s["filename"]: clip for s in samples}, {s["filename"]: runner.sha256(clip) for s in samples}))
    monkeypatch.setattr(runner, "probe_duration", lambda _: 1.0)
    monkeypatch.setattr(runner, "frame_uri", lambda *args: "synthetic-image")
    monkeypatch.setattr(runner, "load_key", lambda _: "synthetic-not-a-key")
    requests = []
    def post(url, body, key):
        requests.append(body)
        return {"id": "synthetic-id", "provider": "Synthetic", "model": runner.CANDIDATES["qwen"][0],
                "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":false}'}}]}
    monkeypatch.setattr(runner, "_post", post)
    monkeypatch.setattr(runner.sys, "argv", argv)
    result = runner.main()
    assert len(requests) == 41 and all("provider" not in body for body in requests)
    assert result["protocol_variant"] == evidence.LEGACY_VARIANT
    assert result["split_context"] is None and result["provider_context"] is None
    assert result["overall"]["total"] == 41 and result["overall"]["errors"] == 0
    assert result["per_event"]["fall"]["total"] == 18 and result["per_event"]["weapon"]["total"] == 23


def test_controlled_entry_still_requires_provider_and_split(tmp_path, monkeypatch):
    _, _, paths = split_fixture(tmp_path)
    samples, manifest, media = paths["test"]
    monkeypatch.setattr(runner, "load_key", lambda _: pytest.fail("must fail before credentials"))
    monkeypatch.setattr(runner.sys, "argv", ["run_eval", "--candidate", "qwen", "--samples", str(samples),
        "--manifest", str(manifest), "--media", str(media)])
    with pytest.raises(SystemExit) as error: runner.main()
    assert error.value.code == 2


@pytest.mark.parametrize("option", ["--provider-config", "--split-manifest", "--split"])
def test_legacy_cannot_be_mixed_with_controlled_options(monkeypatch, option):
    monkeypatch.setattr(runner, "frozen_inputs", lambda *args: pytest.fail("reject incompatible options first"))
    monkeypatch.setattr(runner.sys, "argv", ["run_eval", "--legacy-baseline", "--candidate", "qwen",
        "--samples", "unused.json", "--media", "unused", option, "test"])
    with pytest.raises(SystemExit) as error: runner.main()
    assert error.value.code == 2
