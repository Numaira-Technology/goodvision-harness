"""Window-local PPE batching contracts, without model weights or GPU access."""

import copy
import io
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from goodvision_harness import ppe
from goodvision_harness.videojev_service import WindowError


def crop(value):
    return SimpleNamespace(value=value)


def positive():
    return {"helmet": "missing", "vest": "worn"}, {"evidence": "positive"}


def negative():
    return {"helmet": "worn", "vest": "worn"}, {"evidence": "negative"}


class Batches(ppe._PPE):
    def __init__(self, frames, *, batch_size=4, always_scene=False):
        super().__init__(
            SimpleNamespace(), batch_size=batch_size, always_scene=always_scene
        )
        self.frames = iter(frames)
        self.prepared = weakref.WeakSet()
        self.peak_prepared = 0
        self.batches = []

    def _people(self, image):
        people = next(self.frames)
        if isinstance(people, Exception):
            raise people
        return [
            ({"person_id": f"person{index + 1:02d}"}, value)
            for index, value in enumerate(people)
        ]

    def _prepare_score(self, image, source_time, *, scene):
        class Prepared:
            pass

        if getattr(image, "prepare_error", None):
            raise image.prepare_error
        prepared = Prepared()
        prepared.image, prepared.source_time = image, source_time
        self.prepared.add(prepared)
        self.peak_prepared = max(self.peak_prepared, len(self.prepared))
        return prepared

    def _predict_scores(self, prepared, *, scene, interrupted=None):
        self.batches.append((scene, [item.source_time for item in prepared]))
        for index, item in enumerate(prepared):
            yield index, ({"presence": "absent"}, {}) if scene else item.image.value


def frames(adapter, count=2):
    return adapter._assess_batched_frames(
        [Image.new("RGB", (12, 12)) for _ in range(count)], list(range(count))
    )


def test_preparation_storage_is_bounded_by_batch_size_not_people_or_frames():
    adapter = Batches([[crop(negative()) for _ in range(21)] for _ in range(4)])
    result = frames(adapter, 4)
    assert adapter.peak_prepared <= 4
    assert not adapter.prepared
    assert all(row["detected"] is False for row in result)
    assert all(len(times) <= 4 for _, times in adapter.batches)
    assert any(len(set(times)) > 1 for _, times in adapter.batches)


def test_shared_failure_stops_preparing_later_chunks_and_keeps_received_outcomes():
    failure = RuntimeError("CUDA error: illegal memory access")
    adapter = Batches([
        [crop(positive()), crop(failure), crop(negative()), crop(ValueError("local")),
         crop(negative())],
        [crop(negative())],
    ])
    with pytest.raises(RuntimeError) as caught:
        frames(adapter)
    observation = caught.value.ppe_observation
    assert observation["detected"] is True
    assert adapter.peak_prepared <= 4
    assert len(adapter.batches) == 1
    rows = observation["frames"][0]["people"]
    assert [row["detected"] for row in rows] == [True, None, False, None, None]
    assert rows[3]["error"]["detail"] == "local"
    assert rows[4]["error"]["type"] == "NotAssessed"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_and_serial_aggregation_match_with_unknowns_and_local_errors(batch_size):
    inputs = [
        [crop(negative()), crop(positive()), ValueError("bad crop")],
        [],
        [crop(({"helmet": "unassessable", "vest": "worn"}, {}))],
        ValueError("detector failed"),
    ]
    result = frames(Batches(inputs, batch_size=batch_size), 4)
    assert [row["detected"] for row in result] == [True, False, None, None]
    assert [row["source_time"] for row in result] == [0, 1, 2, 3]
    assert result[0]["people"][0]["person_id"] == "person01"
    assert result[2]["people"][0]["person_id"] == "person01"
    assert result[0]["people"][2]["error"]["detail"] == "bad crop"
    assert result[0]["scene_raw"]["state"] == "not_requested"
    assert result[3]["scene_raw"]["reason"] == "person_detection_incomplete"


def test_local_preparation_error_does_not_drop_other_people():
    adapter = Batches([
        [crop(negative()), SimpleNamespace(prepare_error=ValueError("processor"))],
        [crop(positive())],
    ])
    result = frames(adapter)
    assert [row["detected"] for row in result] == [None, True]
    assert result[0]["people"][1]["error"]["detail"] == "processor"


def native_result(template, choice=0):
    return [[{
        "id": row["id"], "task": "choice", "answer_index": choice,
        "answer": row["options"][choice],
        "probabilities": [0.8 if index == choice else 0.1 for index in range(3)],
    }] for row in template.rows], {"native": True}


@pytest.mark.parametrize("mode", ["batch", "fallback", "missing_method"])
def test_videojev_fallback_and_batch_keep_the_same_attributes_and_templates(mode):
    calls = []

    def prepare(payloads, fps, **kwargs):
        assert payloads[0] == payloads[1] and fps == 1
        assert kwargs["frame_times"][0] == kwargs["frame_times"][1]
        assert not ppe.has_ppe(kwargs["question_template"])
        return object()

    def one(**kwargs):
        calls.append(("one", kwargs["frame_times"]))
        return native_result(kwargs["question_template"])

    def batch(items):
        calls.append(("batch", len(items)))
        if mode == "fallback":
            yield None
            return
        for index in reversed(range(len(items))):
            yield index, native_result(items[index]["question_template"])

    runtime = SimpleNamespace(torch=None, prepare=prepare, predict_prepared=one)
    if mode != "missing_method":
        runtime.predict_prepared_batch = batch
    adapter = ppe.VideoJevPPE(runtime)
    adapter._people = lambda image: [
        ({"person_id": f"person{index}"}, image) for index in range(3)
    ]
    result = frames(adapter)
    assert [row["detected"] for row in result] == [True, True]
    assert [len(row["people"]) for row in result] == [3, 3]
    assert all(row["raw"]["timings"] == {"native": True}
               for frame in result for row in frame["people"])
    if mode == "batch":
        assert calls == [("batch", 4), ("batch", 2)]
    else:
        assert len([call for call in calls if call[0] == "one"]) == 6


@pytest.mark.parametrize("shared", [False, True])
def test_videojev_partial_batch_drains_data_and_falls_back_only_when_safe(shared):
    calls = []
    runtime = SimpleNamespace(torch=None)
    adapter = ppe.VideoJevPPE(runtime)
    good = native_result(adapter.templates[False])
    failure = (
        RuntimeError("CUDA error: illegal memory access")
        if shared else ValueError("bad")
    )

    def batch(items):
        yield 0, failure
        yield 1, copy.deepcopy(good)
        yield 1, ValueError("duplicate must be ignored")
        yield None

    runtime.predict_prepared_batch = batch
    def one(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(good)

    runtime.predict_prepared = one
    interrupted = False
    outcomes = []

    def stopped():
        return interrupted

    for index, result in adapter._predict_scores(
        [{"prepared": index} for index in range(4)], scene=False,
        interrupted=stopped,
    ):
        outcomes.append((index, result))
        if shared and index == 0:
            interrupted = True
    assert [index for index, _ in outcomes] == ([0, 1] if shared else [0, 1, 2, 3])
    assert len(calls) == (0 if shared else 2)


@pytest.mark.parametrize("batch_size", [0, 5, True, 1.5])
def test_invalid_batch_limits_are_rejected(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        ppe._PPE(None, batch_size=batch_size)


class Tensor:
    """CPU arithmetic for testing CLIP's real batch/normalization/ensemble code."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def __truediv__(self, other):
        return Tensor(self.values / other.values)

    def __matmul__(self, other):
        return Tensor(self.values @ other.values)

    def norm(self, p=2, dim=None, keepdim=False):
        return Tensor(np.linalg.norm(self.values, ord=p, axis=dim, keepdims=keepdim))

    def mean(self, dim):
        return Tensor(self.values.mean(axis=dim))

    def t(self):
        return Tensor(self.values.T)

    def to(self, device):
        assert device == "cpu"
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


def clip_encoder():
    calls = {"text": 0, "vision": []}
    text_vectors = np.concatenate([
        np.repeat(np.eye(9)[index][None, :], len(prompts), axis=0)
        for index, prompts in enumerate(
            prompts
            for states in ppe._CLIP_TEXTS.values() for prompts in states.values()
        )
    ])
    image_vectors = np.eye(9)[[1, 3, 4, 5, 3]].copy()
    image_vectors[1:, 7] = 1
    image_vectors[4, 4] = 1  # A helmet tie must remain unknown.

    def processor(*, images=None, text=None, **kwargs):
        if text is not None:
            return {"input_ids": Tensor(text_vectors)}
        return {"pixel_values": Tensor([image_vectors[images.getpixel((0, 0))[0]]])}

    def text_model(input_ids):
        calls["text"] += 1
        return SimpleNamespace(pooler_output=input_ids)

    def vision_model(pixel_values):
        calls["vision"].append(len(pixel_values.values))
        return SimpleNamespace(pooler_output=pixel_values)

    torch = SimpleNamespace(
        inference_mode=nullcontext,
        cat=lambda values, dim=0: Tensor(
            np.concatenate([v.values for v in values], dim)
        ),
        stack=lambda values: Tensor(np.stack([v.values for v in values])),
    )
    model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device="cpu")]),
        text_model=text_model, text_projection=lambda value: value,
        vision_model=vision_model, visual_projection=lambda value: value,
    )
    return SimpleNamespace(torch=torch, processor=processor, model=model), calls


def stable(value):
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items()
                if key not in ("assessment_seconds", "detector_seconds")}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


def test_clip_batch_matches_serial_raw_cosines_states_and_text_cache():
    observed = []
    for size in (1, 4):
        encoder, calls = clip_encoder()
        adapter = ppe.ClipPPE(encoder, batch_size=size)
        adapter._people = lambda image: [
            ({"person_id": f"person{index}"}, Image.new("RGB", (12, 12), (index, 0, 0)))
            for index in (1, 2, 3, 4)
        ]
        result = frames(adapter)
        observed.append(stable(result))
        assert calls["text"] == 1
        assert calls["vision"] == ([1] * 8 if size == 1 else [4, 4])
        assert [row["detected"] for row in result[0]["people"]] == [
            True, False, None, None,
        ]
    assert observed[0] == observed[1]


def test_clip_shared_batch_failure_drains_errors_without_another_forward():
    encoder, calls = clip_encoder()
    adapter = ppe.ClipPPE(encoder)
    adapter._people = lambda image: [({"person_id": str(i)}, image) for i in range(6)]

    def fail(pixel_values):
        calls["vision"].append(len(pixel_values.values))
        raise RuntimeError("CUDA error: illegal memory access")

    encoder.model.vision_model = fail
    with pytest.raises(RuntimeError) as caught:
        frames(adapter)
    rows = caught.value.ppe_observation["frames"][0]["people"]
    assert all(row["error"]["detail"] == "CUDA error: illegal memory access"
               for row in rows[:4])
    assert rows[4]["error"]["type"] == "NotAssessed"
    assert calls["vision"] == [4]


@pytest.mark.parametrize("end", ["missing", "raise", "bad_index"])
def test_incomplete_batch_keeps_received_positive_and_marks_rest_unknown(end):
    adapter = Batches([[crop(negative()) for _ in range(4)], []])

    def predict(prepared, *, scene, interrupted):
        if scene:
            yield 0, ({"presence": "absent"}, {})
            return
        yield 1, positive()
        if end == "raise":
            raise ValueError("truncated batch")
        if end == "bad_index":
            yield -1, negative()

    adapter._predict_scores = predict
    result = frames(adapter)
    assert result[0]["detected"] is True
    assert [row["detected"] for row in result[0]["people"]] == [None, True, None, None]
    assert all("before this item's outcome" in row["error"]["detail"]
               for i, row in enumerate(result[0]["people"]) if i != 1)


@pytest.mark.parametrize("recovery_fails", [False, True])
def test_batch_oom_recovery_retains_cpu_outcomes_and_stops_if_recovery_fails(
    recovery_fails,
):
    recovered = []
    adapter = Batches([
        [crop(RuntimeError("CUDA out of memory")), crop(positive()),
         crop(ValueError("local evidence")), crop(negative()), crop(negative())],
        [],
    ])

    def synchronize():
        recovered.append("sync")
        if recovery_fails:
            raise RuntimeError("synchronize failed")

    adapter.torch.cuda = SimpleNamespace(
        empty_cache=lambda: recovered.append("empty"), synchronize=synchronize,
    )
    if recovery_fails:
        with pytest.raises(WindowError) as caught:
            frames(adapter)
        assert caught.value.code == "cuda_recovery_failed"
        result = caught.value.ppe_observation["frames"]
        assert len(adapter.batches) == 1
    else:
        result = frames(adapter)
        assert len(adapter.batches) == 3  # Full crops, scene tail, person tail.
    assert result[0]["detected"] is True
    assert result[0]["people"][2]["error"]["detail"] == "local evidence"
    assert recovered == ["empty", "sync"]


def test_videojev_invalid_raw_is_retained_and_never_retried():
    runtime = SimpleNamespace(torch=None)
    adapter = ppe.VideoJevPPE(runtime)
    malformed = ([{"id": "ppe/helmet", "probabilities": [float("nan")]}], {})
    runtime.predict_prepared_batch = lambda items: iter([(0, malformed), None])
    runtime.predict_prepared = lambda **kwargs: native_result(adapter.templates[False])
    result = dict(adapter._predict_scores([{}, {}], scene=False))
    assert isinstance(result[0], Exception)
    assert result[0].ppe_raw["raw"] is malformed[0]
    assert result[1][0] == {"helmet": "missing", "vest": "missing"}


def test_clip_nonfinite_diagnostic_remains_unknown_and_preserves_raw():
    encoder, _ = clip_encoder()
    adapter = ppe.ClipPPE(encoder)
    task = ({}, {}, None, False)
    with pytest.raises(ValueError) as caught:
        adapter._format_score([float("nan")] * 9, [0] * len(adapter.texts), scene=False)
    adapter._store_score(task, caught.value)
    assert task[1]["error"]["type"] == "ValueError"
    assert np.isnan(task[1]["raw"]["cosine"]).all()


@pytest.mark.parametrize("first,expected", [(positive(), True), (negative(), None)])
def test_batched_window_preserves_good_frame_and_malformed_source(first, expected):
    adapter = Batches([[crop(first)]])
    adapter._assess_frames = adapter._assess_batched_frames
    with io.BytesIO() as output:
        Image.new("RGB", (12, 12)).save(output, format="JPEG")
        payload = output.getvalue()
    result = adapter.assess([payload, b"invalid jpeg"], [0, 1])
    assert result["aggregation"] == "window_presence"
    assert result["detected"] is expected
    assert result["frames"][1]["detected"] is None
    assert result["frames"][1]["reason"] == "frame_assessment_failed"
    assert [row["source_time"] for row in result["frames"]] == [0, 1]


def test_crop_retention_does_not_grow_across_source_frames():
    adapter = Batches([])
    live = weakref.WeakSet()
    peak = 0

    class Crop:
        value = negative()

    def people(image):
        nonlocal peak
        crops = [Crop() for _ in range(21)]
        live.update(crops)
        peak = max(peak, len(live))
        return [({"person_id": str(i)}, item) for i, item in enumerate(crops)]

    adapter._people = people
    frames(adapter, 4)
    assert peak <= 24  # Current frame plus at most three queued person crops.
    assert not live
