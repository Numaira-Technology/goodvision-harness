"""Candidate contract tests without model weights, CUDA, providers or network."""

import copy
import hashlib
import json

import pytest
from PIL import Image

from tools.benchmark_runtime.vlm_event_policy import annotation_policy
from tools.videojev_probe.contract import (
    LABELS,
    OPTIONS,
    POLICY_SHA,
    VIDEO_SHA,
    parse_choices,
    plan,
    questions,
    select_windows,
)
from tools.videojev_probe.packet import prepare, validate, write_json
from tools.videojev_probe.ppe import merge_equipment, values_for_question
from tools.videojev_probe.preview import preview
from tools.videojev_probe.run import execute, summarize
from tools.videojev_probe.runtime import record_for_grid


def window():
    return {
        "window_id": "W001",
        "frames": [{"media_seconds": 0.4}, {"media_seconds": 0.8166666667}],
        "judgment_time_seconds": 0.8166666667,
    }


def responses(qs, index=1):
    return [
        {
            "id": q["id"],
            "task": "choice",
            "answer_index": index,
            "answer": q["options"][index],
            "probabilities": [0.8 if i == index else 0.1 for i in range(3)],
        }
        for q in qs
    ]


def test_frozen_policy_replacement_and_independent_questions():
    p = plan()
    assert p["execute"] is False
    assert p["event_policy"]["sha256"] == POLICY_SHA
    qs = questions(window())
    assert len(qs) == 6
    for q in qs:
        assert q["task"] == "choice"
        assert q["options"] == OPTIONS
        assert annotation_policy(LABELS)["decision_rules"] in q["query"]
        assert "0.8166666667" in q["query"]
    assert parse_choices(responses(qs, 2), qs) == {q["id"]: None for q in qs}


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.pop(),
        lambda r: r.append(copy.deepcopy(r[0])),
        lambda r: r[0].update(id="P9999/fight"),
        lambda r: r[0].update(answer_index=True),
        lambda r: r[0].update(answer_index=3),
        lambda r: r[0].update(task="grounding"),
        lambda r: r[0].update(probabilities=[float("nan"), 0.8, 0.2]),
        lambda r: r[0].update(probabilities=[-0.1, 0.8, 0.3]),
        lambda r: r[0].update(probabilities=[0.2, 0.8, 0.3]),
        lambda r: r[0].update(probabilities=[0.8, 0.1, 0.1]),
        lambda r: r[0].update(probabilities=[True, 0, 0]),
        lambda r: r[0].update(answer="B. No smoking"),
        lambda r: r[0].update(id=r[1]["id"]),
    ],
)
def test_invalid_response_is_failure_not_a_negative(change):
    qs = questions(window())
    raw = responses(qs)
    change(raw)
    with pytest.raises(ValueError):
        parse_choices(raw, qs)


def snapshots():
    return [
        {
            "state": "ready",
            "stream_session_id": "W001",
            "frame_sequence": i,
            "timestamp_seconds": f["media_seconds"],
            "people": [
                {
                    "track_id": "P0001",
                    "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.9},
                }
            ],
        }
        for i, f in enumerate(window()["frames"])
    ]


def test_yolo_id_supplements_all_scene_questions_without_person_litter():
    qs = questions(window(), snapshots())
    assert len(qs) == 11
    assert "P0001/ground_litter" not in {q["id"] for q in qs}
    assert {f"scene/{label}" for label in LABELS} <= {q["id"] for q in qs}
    assert all("P0001" in q["query"] for q in qs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "detector_failed"),
        ("stream_session_id", "previous-window"),
        ("timestamp_seconds", 0),
        ("frame_sequence", 99),
    ],
)
def test_failed_or_stale_detection_is_not_used(field, value):
    rows = snapshots()
    rows[-1][field] = value
    with pytest.raises(ValueError):
        questions(window(), rows)


def test_person_budget_never_truncates_and_earlier_ids_are_context_only():
    rows = snapshots()
    rows[-1]["people"] = []
    assert len(questions(window(), rows)) == 6
    rows[-1]["people"] = [{"track_id": f"P{i:04}"} for i in range(5)]
    with pytest.raises(ValueError, match="budget"):
        questions(window(), rows)


def test_uniform_selection_is_label_blind_and_bounded():
    rows = [{"window_id": f"W{i:03}"} for i in range(69)]
    selected = select_windows(rows)
    assert len(selected) == len({r["window_id"] for r in selected}) == 12
    assert selected[0] == rows[0] and selected[-1] == rows[-1]
    for count in (0, 13, True):
        with pytest.raises(ValueError):
            select_windows(rows, count)


def test_native_grid_cannot_hide_frame_resampling():
    class Processor:
        def _calculate_timestamps(self, ids, fps, factor):
            assert ids == [0, 1, 2] and fps == 2.5 and factor == 2
            return [0.2, 0.8]

    record = record_for_grid(Processor(), [2, 32, 32], 3, 2.5)
    assert record["frames"] == 3 and record["tokens"] == 512
    with pytest.raises(ValueError, match="resampled"):
        record_for_grid(Processor(), [3, 32, 32], 3, 2.5)


@pytest.fixture
def packet(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    policy = annotation_policy(LABELS)
    windows, rows = [], []
    for i in range(13):
        wid = f"W{i + 1:03}"
        (source / wid).mkdir()
        frames = []
        for j in range(2):
            path = f"{wid}/{j}.jpg"
            Image.new("RGB", (32, 32), (i * 12, j * 100, 90)).save(source / path)
            frames.append(
                {
                    "path": path,
                    "media_seconds": i + j * 0.4,
                    "sha256": hashlib.sha256((source / path).read_bytes()).hexdigest(),
                }
            )
        windows.append(
            {
                "window_id": wid,
                "frames": frames,
                "judgment_time_seconds": frames[-1]["media_seconds"],
            }
        )
        rows.append(
            {
                "window_id": wid,
                "evidence_timestamps_seconds": [f["media_seconds"] for f in frames],
                "judgment_time_seconds": frames[-1]["media_seconds"],
                "judgments": {
                    k: {"detected": False, "reason": "fixture only"} for k in LABELS
                },
            }
        )
    manifest = {
        "windows": windows,
        "source_video_sha256": VIDEO_SHA,
        "policy_sha256": POLICY_SHA,
        "policy_version": policy["version"],
        "source_duration_seconds": 20,
        "image_side": 720,
        "jpeg_quality": 90,
        "sampling_fps": 2.5,
        "prediction_data_included": False,
        "previous_annotations_included": False,
    }
    reference = {
        "source_video_sha256": VIDEO_SHA,
        "policy_sha256": POLICY_SHA,
        "human_reviewed": False,
        "annotator_model": "test-fixture-not-astra",
        "rows": rows,
        "inspected_window_ids": [w["window_id"] for w in windows],
        "uninspected_window_ids": [],
    }
    write_json(source / "input-manifest.json", manifest)
    write_json(source / "event-policy.json", policy)
    write_json(source / "annotations.json", reference)
    dest = tmp_path / "packet"
    prepare(source, dest)
    return dest


def test_packet_preserves_exact_images_and_does_not_leak_reference(packet):
    manifest, _policy, reference, images = validate(packet)
    assert len(manifest["windows"]) == len(reference["rows"]) == len(images) == 12
    requests = (packet / "scene-questions.json").read_text()
    assert "fixture only" not in requests and "test-fixture-not-astra" not in requests
    for w in manifest["windows"]:
        for f, data in zip(w["frames"], images[w["window_id"]], strict=True):
            assert hashlib.sha256(data).hexdigest() == f["sha256"]
    with pytest.raises(FileExistsError):
        prepare(packet, packet)


def test_tampered_frame_rejected_before_model_load(packet, tmp_path):
    frame = next(packet.glob("W*/*.jpg"))
    frame.write_bytes(b"changed")

    def forbidden(_):
        pytest.fail("model must not load for invalid packet")

    with pytest.raises(ValueError, match="fingerprint"):
        execute(packet, tmp_path / "out", "cache", runtime_factory=forbidden)


def test_sequential_probe_records_failed_windows_and_preserves_denominator(
    packet, tmp_path
):
    class Fake:
        count = 0

        def __init__(self, _):
            self.metadata = {"mock": True}

        def predict(self, images, fps, qs):
            self.count += 1
            assert len(images) == 2 and fps == 2.5
            return (responses(qs) if self.count != 2 else []), {"mock": True}

    out = tmp_path / "out"
    summary = execute(packet, out, "cache", runtime_factory=Fake)
    assert summary["planned_windows"] == 12
    assert summary["complete_windows"] == 11
    assert summary["complete_rate"] == 11 / 12
    assert summary["astra_agreement"] == 1
    assert summary["decided_coverage_of_all_reference_certain"] == 11 / 12
    assert summary["person_accuracy"] is None
    assert json.loads((out / "W001-request.json").read_text())[0]["task"] == "choice"


def test_missing_gpu_writes_failure_without_fabricating_latency(packet, tmp_path):
    def no_gpu(_):
        raise RuntimeError("no CUDA")

    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="no CUDA"):
        execute(packet, out, "cache", runtime_factory=no_gpu)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["not_attempted_windows"] == 12
    assert summary["latency_p50_seconds"] is None
    assert summary["astra_agreement"] is None
    assert json.loads((out / "manifest.json").read_text())["state"] == "failed"


def test_uncertainty_and_failures_not_counted_as_agreement(packet):
    manifest, _, ref, _ = validate(packet)
    decisions = {f"scene/{label}": None for label in LABELS}
    rows = [
        {
            "window_id": manifest["windows"][0]["window_id"],
            "complete": True,
            "decisions": decisions,
            "probe_window_seconds": 2.0,
        }
    ]
    s = summarize(manifest["windows"], rows, ref)
    assert s["complete_rate"] == s["within_3s_rate"] == 1 / 12
    assert s["astra_agreement"] is None
    assert s["decided_coverage_of_all_reference_certain"] == 0


def test_nonfinite_gpu_output_is_logged_as_failure_and_does_not_break_report(
    packet, tmp_path
):
    class NumericalFailure:
        def __init__(self, _):
            self.metadata = {"mock": True}

        def predict(self, _images, _fps, qs):
            raw = responses(qs)
            raw[0]["probabilities"][0] = float("nan")
            return raw, {}

    out = tmp_path / "out"
    s = execute(packet, out, "cache", runtime_factory=NumericalFailure)
    assert s["attempted_windows"] == 12
    assert s["complete_windows"] == 0
    assert s["astra_agreement"] is None
    row = json.loads((out / "W001-result.json").read_text())
    assert row["error"] == "ValueError" and "nan" in row["invalid_output_repr"]


def test_fatal_gpu_failure_aborts_without_removing_unattempted_windows(
    packet, tmp_path
):
    class GPUFailure:
        def __init__(self, _):
            self.metadata = {"mock": True}

        def predict(self, *_):
            raise RuntimeError("mock CUDA error")

    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="remaining windows"):
        execute(packet, out, "cache", runtime_factory=GPUFailure)
    s = json.loads((out / "summary.json").read_text())
    assert s["attempted_windows"] == 1 and s["not_attempted_windows"] == 11
    assert s["planned_windows"] == 12 and s["complete_rate"] == 0


def test_changed_sampling_clock_and_person_scope_without_detector_fail_closed(
    packet, tmp_path
):
    with pytest.raises(ValueError, match="YOLO"):
        execute(packet, tmp_path / "out", "cache", scope="person")
    path = packet / "input-manifest.json"
    data = json.loads(path.read_text())
    data["sampling_fps"] = 1
    write_json(path, data)
    with pytest.raises(ValueError, match=r"2\.5 fps"):
        validate(packet)


@pytest.mark.parametrize("variant", ["baseline", "focused", "equipment"])
@pytest.mark.parametrize("order", ["canonical", "rotate", "reverse"])
def test_ppe_variants_keep_other_five_queries_and_policy_identical(variant, order):
    from tools.benchmark_runtime.event_sensitivity import PPE_RULE

    original = questions(window(), snapshots())
    qs = questions(window(), snapshots(), ppe_variant=variant, ppe_option_order=order)
    others = [q for q in original if not q["id"].endswith("/ppe_violation")]
    assert [q for q in qs if "/ppe_" not in q["id"]] == others
    assert len(qs) == (13 if variant == "equipment" else 11)
    assert plan()["event_policy"]["sha256"] == POLICY_SHA
    for q in qs:
        if "/ppe_" not in q["id"]:
            continue
        assert PPE_RULE in q["query"]
        assert "0.8166666667" in q["query"]
        assert "P0001" in q["query"]
        if variant != "baseline":
            assert "Fighting sensitivity" not in q["query"]
            assert "ground_litter:" not in q["query"]
            assert "office" in q["query"]
            assert len(q["query"]) < len(original[4]["query"])


@pytest.mark.parametrize("variant", ["baseline", "focused", "equipment"])
@pytest.mark.parametrize(
    "order,expected",
    [
        ("canonical", [True, False, None]),
        ("rotate", [False, None, True]),
        ("reverse", [None, False, True]),
    ],
)
def test_permutation_decodes_semantics_not_position(variant, order, expected):
    qs = questions(window(), ppe_variant=variant, ppe_option_order=order)
    for index in range(3):
        decoded = parse_choices(responses(qs, index), qs)
        for q in qs:
            assert decoded[q["id"]] is (
                expected[index] if "/ppe_" in q["id"] else [True, False, None][index]
            )


@pytest.mark.parametrize(
    "head,torso,want",
    [
        (True, True, True),
        (True, False, True),
        (True, None, True),
        (False, True, True),
        (False, False, False),
        (False, None, None),
        (None, True, True),
        (None, False, None),
        (None, None, None),
    ],
)
def test_equipment_rule_missing_either_and_uncertainty(head, torso, want):
    original = {
        "scene/ppe_head": head,
        "scene/ppe_torso": torso,
        "scene/fight": False,
        "P0001/ppe_head": False,
        "P0001/ppe_torso": False,
    }
    result = merge_equipment(original)
    assert result == {
        "scene/ppe_violation": want,
        "scene/fight": False,
        "P0001/ppe_violation": False,
    }
    assert original["scene/ppe_head"] is head  # Observation evidence not mutated.


def test_missing_or_invalid_equipment_never_becomes_compliant():
    with pytest.raises(ValueError, match="incomplete"):
        merge_equipment({"scene/ppe_head": False})
    with pytest.raises(ValueError, match="conflicting"):
        merge_equipment(
            {
                "scene/ppe_head": True,
                "scene/ppe_torso": False,
                "scene/ppe_violation": False,
            }
        )
    with pytest.raises(ValueError, match="invalid"):
        merge_equipment({"scene/ppe_head": 0, "scene/ppe_torso": False})
    qs = questions(window(), ppe_variant="equipment")
    raw = responses(qs)
    raw.pop(5)  # Torso response lost: whole window must fail, not partial success.
    with pytest.raises(ValueError, match="incomplete"):
        parse_choices(raw, qs)


def test_equipment_runner_keeps_raw_observations_and_six_class_summary(
    packet, tmp_path
):
    class Fake:
        def __init__(self, _):
            self.metadata = {"mock": True}

        def predict(self, images, fps, qs):
            assert len(images) == 2 and fps == 2.5 and len(qs) == 7
            raw = responses(qs, 1)
            for q, response in zip(qs, raw, strict=True):
                values = values_for_question(q, OPTIONS)
                value = True if q["id"].endswith("/ppe_head") else False
                index = next(i for i, v in enumerate(values) if v is value)
                response.update(
                    answer_index=index,
                    answer=q["options"][index],
                    probabilities=[0.8 if i == index else 0.1 for i in range(3)],
                )
            return raw, {"mock": True}

    out = tmp_path / "new-run"
    s = execute(
        packet,
        out,
        "unused",
        ppe_variant="equipment",
        ppe_option_order="rotate",
        runtime_factory=Fake,
    )
    assert s["complete_windows"] == 12
    assert s["per_label"]["ppe_violation"]["false_positive"] == 12
    row = json.loads((out / "W001-result.json").read_text())
    assert len(row["decisions"]) == 6 and len(row["observations"]) == 7
    assert row["decisions"]["scene/ppe_violation"] is True
    assert row["observations"]["scene/ppe_torso"] is False
    assert row["option_semantics"]["scene/ppe_head"] == [False, None, True]
    assert s["ppe_experiment"]["criteria_changed"] is False


@pytest.fixture
def saved_person_run(packet, tmp_path):
    from tools.videojev_probe.packet import digest

    root = tmp_path / "old-person-run"
    root.mkdir()
    write_json(
        root / "manifest.json",
        {
            "state": "finished",
            "probe_scope": "person",
            "input_hashes": {
                n: digest(packet / n)
                for n in (
                    "input-manifest.json",
                    "event-policy.json",
                    "annotations.json",
                )
            },
        },
    )
    manifest, _, _, _ = validate(packet)
    for w in manifest["windows"]:
        ss = [
            {
                **s,
                "stream_session_id": w["window_id"],
                "timestamp_seconds": f["media_seconds"],
            }
            for s, f in zip(snapshots(), w["frames"], strict=True)
        ]
        write_json(
            root / f"{w['window_id']}-result.json",
            {
                "window_id": w["window_id"],
                "complete": True,
                "detector_snapshots": ss,
            },
        )
    return root


def test_offline_preview_uses_saved_ids_without_model_or_network(
    packet, saved_person_run, tmp_path, monkeypatch
):
    import socket

    from tools.videojev_probe import runtime

    def forbidden(*_, **__):
        pytest.fail("offline preview must not load a model or access network")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(runtime, "VideoJevRuntime", forbidden)
    out = tmp_path / "preview"
    result = preview(
        packet,
        out,
        ppe_variant="equipment",
        ppe_option_order="reverse",
        person_run=saved_person_run,
    )
    assert result["execute"] is False
    assert set(result["question_counts"].values()) == {13}
    requests = (out / "requests.json").read_text()
    assert "fixture only" not in requests and "test-fixture-not-astra" not in requests
    assert "P0001/ppe_head" in requests
    with pytest.raises(FileExistsError):
        preview(packet, out)


@pytest.mark.parametrize("tamper", ["fingerprint", "coordinate", "frame", "session"])
def test_replay_rejects_bad_evidence_before_model_load(
    packet, saved_person_run, tmp_path, tamper
):
    if tamper == "fingerprint":
        p = saved_person_run / "manifest.json"
        d = json.loads(p.read_text())
        d["input_hashes"]["event-policy.json"] = "wrong"
    else:
        p = saved_person_run / "W001-result.json"
        d = json.loads(p.read_text())
        s = d["detector_snapshots"][0]
        if tamper == "coordinate":
            s["people"][0]["bbox"]["x1"] = -0.1
        elif tamper == "frame":
            s["timestamp_seconds"] = 999
        else:
            s["stream_session_id"] = "other"
    write_json(p, d)

    def forbidden(_):
        pytest.fail("invalid replay must fail before GPU initialization")

    with pytest.raises(ValueError):
        execute(
            packet,
            tmp_path / "out",
            "cache",
            scope="person",
            person_run=saved_person_run,
            runtime_factory=forbidden,
        )


def test_unknown_experiment_rejected_before_model_and_replay_timing_labeled(
    packet, saved_person_run, tmp_path
):
    def forbidden(_):
        pytest.fail("invalid profile must fail before GPU initialization")

    with pytest.raises(ValueError, match="unknown PPE"):
        execute(
            packet,
            tmp_path / "bad",
            "cache",
            ppe_variant="typo",
            runtime_factory=forbidden,
        )

    class Fake:
        def __init__(self, _):
            self.metadata = {"mock": True}

        def predict(self, images, fps, qs):
            return responses(qs), {}

    result = execute(
        packet,
        tmp_path / "replay",
        "cache",
        scope="person",
        person_run=saved_person_run,
        runtime_factory=Fake,
    )
    assert result["complete_windows"] == 12
    assert "excludes live detection" in result["scope"]
    assert result["detector_replay"]["manifest_sha256"]
