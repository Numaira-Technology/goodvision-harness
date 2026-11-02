"""PPE evidence and failure isolation without weights, GPU or network access."""

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_vlm_independent import jpeg

from goodvision_harness import ppe
from goodvision_harness.videojev_contract import OPTIONS


def person(helmet="worn", vest="worn", **extra):
    return {"presence": "present", "helmet": helmet, "vest": vest, **extra}


@pytest.mark.parametrize(
    "people,presence,detector,expected",
    [
        ([], "absent", "completed", False),
        ([], "present", "completed", None),
        ([], "unclear", "completed", None),
        ([], "absent", "failed", None),
        ([person()], "present", "completed", False),
        ([person(helmet="unassessable")], "present", "completed", None),
        ([person(vest="unassessable")], "present", "completed", None),
        ([person(), person(helmet="missing")], "absent", "completed", True),
        ([person(helmet="missing", vest="unassessable")], "present", "completed", True),
        ([person(error={"type": "failed"})], "present", "completed", None),
    ],
)
def test_frame_aggregation_preserves_three_states(people, presence, detector, expected):
    assert ppe._merge_frame(people, presence, detector)["detected"] is expected


class Assessment(ppe._PPE):
    def __init__(self, frames):
        super().__init__(SimpleNamespace())
        self.frames = iter(frames)

    def _assess_frame(self, image, source_time):
        value = next(self.frames)
        if isinstance(value, Exception):
            raise value
        return {"source_time": source_time, "detected": value}


@pytest.mark.parametrize(
    "frames,expected",
    [
        ([False, False], False),
        ([None, False], None),
        ([True, None], True),
        ([ValueError("bad crop"), True], True),
        ([False, ValueError("bad crop")], None),
    ],
)
def test_both_source_frames_contribute_and_local_failure_remains_unknown(
    frames, expected
):
    result = Assessment(frames).assess([jpeg(), jpeg()], [1, 1.5])
    assert result["detected"] is expected
    assert [frame["source_time"] for frame in result["frames"]] == [1, 1.5]


@pytest.mark.parametrize("times", [[1, 1], [2, 1], [0, float("nan")], [True, 2], [1]])
def test_invalid_source_times_never_reach_assessment(times):
    with pytest.raises(ValueError, match="distinct source frames"):
        Assessment([]).assess([jpeg(), jpeg()], times)


def detector(monkeypatch, tmp_path, boxes):
    weights = tmp_path / "staged.pt"
    weights.write_bytes(b"offline fake, never loaded")
    monkeypatch.setenv("DETECTOR_WEIGHTS", str(weights))
    monkeypatch.setenv("DETECTOR_CONF", "0.35")
    fake = SimpleNamespace(
        predict=lambda *a, **kw: [
            SimpleNamespace(
                boxes=[
                    SimpleNamespace(xyxy=np.array([box]), conf=np.array([0.9]))
                    for box in boxes
                ]
            )
        ]
    )
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: fake)
    )
    return ppe._PPE(SimpleNamespace())


def test_failed_crop_does_not_erase_other_persons_positive(monkeypatch, tmp_path):
    adapter = detector(monkeypatch, tmp_path, [(0, 0, 20, 40), (50, 0, 90, 100)])
    original = ppe._square_pad

    def pad(image):
        if image.width < 30:
            raise ValueError("one crop failed")
        return original(image)

    monkeypatch.setattr(ppe, "_square_pad", pad)
    adapter._score = lambda *a, scene, **kw: (
        {"presence": "present"} if scene else {"helmet": "missing", "vest": "worn"},
        {},
    )
    result = adapter._assess_frame(Image.new("RGB", (100, 100)), 1)
    assert result["detected"] is True
    assert [row["detected"] for row in result["people"]] == [None, True]
    assert result["people"][0]["error"]["detail"] == "one crop failed"
    assert result["people"][1]["crop_bbox_pixels"] == [48, 0, 92, 100]


@pytest.mark.parametrize("box", [(-1, 0, 20, 40), (0, 0, 110, 40), (20, 0, 10, 40)])
def test_invalid_person_box_cannot_become_an_empty_negative(monkeypatch, tmp_path, box):
    adapter = detector(monkeypatch, tmp_path, [box])
    adapter._score = lambda *a, **kw: ({"presence": "absent"}, {})
    result = adapter._assess_frame(Image.new("RGB", (100, 100)), 1)
    assert result["detector"]["state"] == "failed"
    assert result["detected"] is None


def test_derived_ppe_does_not_replace_native_scores_or_other_labels():
    expected = [{"id": name, "options": OPTIONS} for name in (ppe.PPE_ID, "scene/fall")]
    raw = [
        {
            "id": row["id"],
            "task": "choice",
            "answer_index": 0,
            "answer": OPTIONS[0],
            "probabilities": [0.8, 0.1, 0.1],
        }
        for row in expected
    ]
    original = copy.deepcopy(raw)
    ppe.attach_observation(raw, {"detected": None, "frames": []})
    assert raw[1] == original[1]
    assert {
        key: value for key, value in raw[0].items() if key != "derived"
    } == original[0]
    assert ppe.parse_ppe_choices(raw, expected) == {
        ppe.PPE_ID: None,
        "scene/fall": True,
    }
    raw[0]["probabilities"] = ["nan"]
    raw[0]["derived"]["detected"] = False
    assert ppe.parse_ppe_choices(raw, expected)[ppe.PPE_ID] is False
    raw[1]["probabilities"] = ["nan"]
    with pytest.raises(ValueError):
        ppe.parse_ppe_choices(raw, expected)


def test_shared_gpu_failure_retains_prior_frame_evidence():
    adapter = Assessment([True, RuntimeError("CUDA error: illegal memory access")])
    with pytest.raises(RuntimeError) as caught:
        adapter.assess([jpeg(), jpeg()], [1, 2])
    assert caught.value.ppe_observation["detected"] is True
    assert caught.value.ppe_observation["frames"] == [
        {"source_time": 1, "detected": True}
    ]


def test_recoverable_crop_oom_keeps_remaining_people(monkeypatch, tmp_path):
    adapter = detector(monkeypatch, tmp_path, [(0, 0, 20, 40), (50, 0, 90, 100)])
    recovered, crops = [], []
    adapter.torch.cuda = SimpleNamespace(
        empty_cache=lambda: recovered.append("empty"),
        synchronize=lambda: recovered.append("sync"),
    )

    def score(image, source_time, *, scene):
        if scene:
            return {"presence": "present"}, {}
        crops.append(source_time)
        if len(crops) == 1:
            raise RuntimeError("CUDA out of memory")
        return {"helmet": "missing", "vest": "worn"}, {}

    adapter._score = score
    result = adapter._assess_frame(Image.new("RGB", (100, 100)), 1)
    assert result["detected"] is True
    assert [row["detected"] for row in result["people"]] == [None, True]
    assert recovered == ["empty", "sync"]


def test_native_ppe_failure_keeps_cpu_scores_and_stops_poisoned_worker():
    from goodvision_harness.videojev_runtime import VideoJevRuntime
    from goodvision_harness.videojev_service import Scorer, WindowError
    from goodvision_harness.vlm_questions import QuestionTemplate

    error = WindowError("cuda_recovery_failed", "synchronize failed")

    def fail(*_):
        raise error

    runtime = object.__new__(VideoJevRuntime)
    runtime._ppe = SimpleNamespace(assess=fail)
    prepared = SimpleNamespace(
        question_template=QuestionTemplate(({"id": ppe.PPE_ID},)),
        source_payloads=(jpeg(), jpeg()),
        frame_times=(1, 2),
    )
    raw = [
        {"id": ppe.PPE_ID, "probabilities": [0.8, 0.1, 0.1]},
        {"id": "scene/fall", "probabilities": [0.1, 0.8, 0.1]},
    ]
    with pytest.raises(ppe.PPEFailure) as caught:
        runtime._append_ppe(raw, {"decision_seconds": 1}, prepared)
    scorer = object.__new__(Scorer)
    stopped = []
    scorer._stop_shared = stopped.append
    result = scorer._gpu_error(caught.value)
    assert stopped == [result]
    assert result.code == "cuda_shared_failure"
    assert result.raw is raw and raw[0]["derived"]["detected"] is None
    assert raw[1] == {"id": "scene/fall", "probabilities": [0.1, 0.8, 0.1]}
    assert result.timings["decision_seconds"] == 1
    assert result.timings["ppe_seconds"] >= 0


def test_native_crop_scoring_uses_one_source_frame_without_recursive_ppe():
    calls = []

    def prepare(payloads, fps, **kwargs):
        calls.append((payloads, fps, kwargs))
        return object()

    def predict(prepared, *, question_template, frame_times):
        assert not ppe.has_ppe(question_template)
        return [
            [
                {
                    "id": row["id"],
                    "task": "choice",
                    "answer_index": 2,
                    "answer": row["options"][2],
                    "probabilities": [0.1, 0.1, 0.8],
                }
            ]
            for row in question_template.rows
        ], {}

    runtime = SimpleNamespace(torch=None, prepare=prepare, predict_prepared=predict)
    values, raw = ppe.VideoJevPPE(runtime)._score(
        Image.new("RGB", (80, 160)),
        3.5,
        scene=False,
    )
    payloads, fps, kwargs = calls[0]
    assert payloads[0] == payloads[1] and fps == 1
    assert kwargs["frame_times"] == (3.5, 3.5)
    assert values == {"helmet": "unassessable", "vest": "unassessable"}
    assert raw["input_mode"] == "one_source_frame_copied_twice"
