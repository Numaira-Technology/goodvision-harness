"""Per-video feature selection for the offline runner."""

from __future__ import annotations

import argparse
import json

import pytest
from bench.schema import Question, question_target_label
from bench.vocabulary import (
    check_supported,
    load_definitions,
    required_labels,
    session_features,
)


def _q(qa_id: str, capability: str, context: dict, **raw) -> Question:
    payload = {
        "qa_id": qa_id,
        "capability": capability,
        "question": "?",
        "video_id": "v1",
        "answer_type": "text",
        "public_context": context,
        **raw,
    }
    return Question.from_json(payload)


def test_event_and_temporal_questions_read_their_own_context_key():
    recognition = _q(
        "a", "event_recognition", {"queried_event_label": "person_sits_down"},
    )
    temporal = _q("b", "temporal_questions", {"event_label": "person_sits_down"})
    assert question_target_label(recognition) == "person_sits_down"
    assert question_target_label(temporal) == "person_sits_down"


def test_the_benchmark_conflict_label_resolves_to_the_built_in_event():
    """physical_conflict is scored against the harness `fight` event, so the
    session has to enable `fight` rather than the benchmark's own name."""

    question = _q(
        "a", "event_recognition", {"queried_event_label": "physical_conflict"},
    )
    assert question_target_label(question) == "fight"
    assert session_features([question]) == ("fight",)


def test_counting_and_location_questions_name_no_event():
    for capability in ("counting", "location_tracking"):
        assert question_target_label(_q("a", capability, {})) is None


def test_a_video_with_no_event_questions_keeps_the_harness_default():
    """Returning an empty tuple would be rejected as an empty feature list,
    and would also change how earlier runs captured these videos."""

    questions = [_q("a", "counting", {}), _q("b", "location_tracking", {})]
    assert session_features(questions) is None


def test_one_session_enables_only_what_its_video_is_asked_about():
    questions = [
        _q("a", "event_recognition", {"queried_event_label": "person_texts_on_phone"}),
        _q("b", "temporal_questions", {"event_label": "person_texts_on_phone"}),
        _q("c", "event_recognition", {"queried_event_label": "person_opens_trunk"}),
        _q("d", "counting", {}),
    ]
    assert session_features(questions) == (
        "person_opens_trunk", "person_texts_on_phone",
    )


def test_a_target_label_outside_the_context_is_still_found():
    question = _q("a", "event_recognition", {}, target_label="person_purchases")
    assert question_target_label(question) == "person_purchases"


def test_required_labels_spans_every_video_in_the_run():
    questions = [
        _q("a", "event_recognition", {"queried_event_label": "person_sits_down"}),
        _q("b", "event_recognition", {"queried_event_label": "fight"}),
    ]
    assert required_labels(questions) == ("fight", "person_sits_down")


def test_an_undeclared_label_names_itself_and_the_fix():
    with pytest.raises(RuntimeError) as error:
        check_supported(["fight", "person_sits_down"], ["fight", "smoking"])
    message = str(error.value)
    assert "person_sits_down" in message
    assert "fight" not in message.split(":", 1)[1].split(".", 1)[0]
    assert "events.definitions" in message


def test_a_fully_cached_rerun_never_contacts_the_harness(tmp_path, monkeypatch):
    """Answering from saved results starts no session, so an unreachable or
    unconfigured harness must not abort the command."""

    from bench import run_bench

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({
            "qa_id": "q1", "capability": "event_recognition", "question": "?",
            "video_id": "v1", "answer_type": "text",
            "public_context": {"queried_event_label": "person_sits_down"},
        }) + "\n",
        encoding="utf-8",
    )
    media = tmp_path / "media.jsonl"
    media.write_text(json.dumps({"video_id": "v1", "path": "v1.mp4"}) + "\n", "utf-8")

    output = tmp_path / "out"
    directory = output / "videos" / "video-v1"
    directory.mkdir(parents=True)
    public = [{
        "qa_id": "q1", "question": "?", "capability": "event_recognition",
        "answer_type": "text",
        "public_context": {"queried_event_label": "person_sits_down"},
    }]
    (directory / "result.json").write_text(json.dumps({
        "status": "complete", "records_downloaded": True,
        "metadata": {"questions": public},
        "predictions": [{"qa_id": "q1", "prediction": "yes"}],
    }), encoding="utf-8")

    def unreachable(*args, **kwargs):
        raise AssertionError("a cached rerun must not call the harness")

    monkeypatch.setattr(run_bench, "supported_labels", unreachable)
    monkeypatch.setattr(run_bench, "capture_offline", unreachable)

    args = argparse.Namespace(
        questions=questions, media=media, output=None, output_dir=output,
        base_url="http://unreachable.invalid", media_root=str(tmp_path),
        reuse_results=True, reanswer=False, no_auto_features=False,
        observation_max_tokens=None, answer_max_tokens=None,
    )
    assert run_bench.run_offline(args) == 0
    rows = (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[0]) == {"qa_id": "q1", "prediction": "yes"}


def test_a_fully_declared_run_passes_preflight():
    supported = [definition["label"] for definition in load_definitions()] + ["fight"]
    assert check_supported(["fight", "person_purchases"], supported) is None
