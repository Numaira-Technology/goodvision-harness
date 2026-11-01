"""Native D3 contracts and batching with CPU fakes, no model downloads."""

from dataclasses import dataclass, field
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from goodvision_harness.decision3_model import Decision3Inspector
from goodvision_harness.decision3_runtime import MODEL, REVISION, Decision3Runtime, _PreparedVideo
from goodvision_harness.videojev_runtime import VideoJevRuntime, _PreparedVideo as VideoJevPrepared
from goodvision_harness.vlm_questions import QuestionTemplate


class Array(np.ndarray):
    def __new__(cls, value, dtype=None, **kwargs):
        return np.asarray(value, dtype=dtype).view(cls)

    def nonzero(self):
        return np.argwhere(np.asarray(self))

    def index_select(self, dim, indices):
        return Array(np.take(self, indices, axis=dim))


OPTIONS = ["Present", "Absent", "Uncertain"]
ROWS = tuple({"id": "scene/" + label, "options": OPTIONS} for label in ("fight", "smoking"))


def answer(index=0):
    return {"type": "choice", "choice": OPTIONS[index], "confidence": 1.0,
            "probabilities": {option: float(i == index) for i, option in enumerate(OPTIONS)}}


def test_partial_official_errors_keep_valid_sibling_verdicts():
    inspector = object.__new__(Decision3Inspector)
    inspector.last_http_timing = {}
    response = {"model": MODEL, "revision": REVISION, "raw": {"answers": {
        "scene/fight": answer(), "scene/smoking": {"type": "choice", "error": "inference_failed"},
    }}}
    verdicts = inspector._decode(response, ROWS, ("fight", "smoking"))
    assert [v.detected for v in verdicts] == [True, None]
    assert "decision3_question_error" in verdicts[1].grounding_issues
    assert list(inspector.last_question_errors) == ["scene/smoking"]
    with pytest.raises(ValueError, match="unexpected"):
        inspector._decode({**response, "revision": "other"}, ROWS, ("fight", "smoking"))


def test_warmup_cannot_mark_an_all_error_model_ready():
    runtime = object.__new__(Decision3Runtime)
    raw = {"answers": {row["id"]: answer() for row in ROWS}}
    runtime.validate_warmup(raw, ROWS)
    raw["answers"]["scene/fight"] = {"type": "choice", "error": "max_length_exceeded"}
    with pytest.raises(ValueError, match="question error"):
        runtime.validate_warmup(raw, ROWS)


def test_videojev_prepared_binding_tracks_runtime_template_times_and_cache_mode():
    calls = []
    runtime = object.__new__(VideoJevRuntime)
    runtime.prefix_cache = True
    runtime.compile_questions = lambda rows: SimpleNamespace(
        bind=lambda record, times, context, **kwargs: calls.append((rows, times, kwargs)) or ("group",),
    )
    template = QuestionTemplate(ROWS)
    video = VideoJevPrepared(None, {"tokens": 10}, 1000, {"preprocess_seconds": 0.0})
    bound = runtime._bind(video, template, [0.0, 0.5])
    assert bound.question_groups == ("group",)
    assert runtime._bind(bound, template, (0.0, 0.5)) is bound
    assert len(calls) == 1
    changed = runtime._bind(bound, template, [1.0, 1.5])
    runtime.prefix_cache = False
    changed = runtime._bind(changed, template, [1.0, 1.5])
    runtime._bind(changed, QuestionTemplate(ROWS), [1.0, 1.5])
    assert len(calls) == 4
    other = object.__new__(VideoJevRuntime)
    other.prefix_cache = False
    other.compile_questions = runtime.compile_questions
    assert other._bind(changed, template, [1.0, 1.5]).question_runtime is other
    assert len(calls) == 5


def test_d3_compiles_complete_prompts_and_reuses_only_identical_bindings():
    from goodvision_harness.decision3_runtime import _CompiledQuestions

    tokenized = []
    def tokenizer(texts, **kwargs):
        tokenized.append(texts)
        return {"input_ids": [[ord(char) for char in text] for text in texts]}
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        get_text_with_replacements=lambda texts, _images, videos: ([texts[0].replace("<video>", videos[0])], None),
        create_mm_token_type_ids=lambda tokens: np.zeros_like(tokens),
    )
    runtime = SimpleNamespace(
        shared_window_state=True, torch=SimpleNamespace(tensor=Array, as_tensor=Array, long=np.int64),
        native=SimpleNamespace(processor=processor, max_length=1000,
                               video_text=lambda state, rendered, *_: "<video>" + rendered),
        helpers=SimpleNamespace(IMAGE_TOKEN="<image>", VIDEO_TOKEN="<video>", normalize_question=lambda row: SimpleNamespace(
            rendered=row["instructions"], kind="choice", keys=tuple(row["criteria"]),
        )),
    )
    rows = tuple({**row, "query_parts": ("Source ", " " + row["id"])} for row in ROWS)
    compiled = _CompiledQuestions(runtime, rows)
    media = SimpleNamespace(video_texts=["pixels"])
    first = compiled.bind(media, "[0,0.5]")
    assert tokenized == [["pixelsSource [0,0.5] scene/fight", "pixelsSource [0,0.5] scene/smoking"]]
    assert first[-1] == len("pixelsSource [0,0.5] scene/")
    assert compiled.bind(media, "[0,0.5]") is first
    assert len(tokenized) == 1
    compiled.bind(media, "[1,1.5]")
    compiled.bind(SimpleNamespace(video_texts=["other pixels"]), "[1,1.5]")
    assert len(tokenized) == 3 and len(compiled.cache) == 2
    runtime.shared_window_state = False
    assert compiled.bind(media, "[1,1.5]")[-1] == 0
    assert len(tokenized) == 4


def test_shared_question_tails_copy_all_window_cache_state():
    class Cache:
        def __init__(self):
            self.keys = Array([[10], [20]])
            self.conv = Array([[30], [40]])
            self.recurrent = Array([[50], [60]])

        def reorder_cache(self, indices):
            for name in ("keys", "conv", "recurrent"):
                setattr(self, name, getattr(self, name)[indices].copy())

    cache, calls = Cache(), []
    def language(**kwargs):
        working = kwargs["past_key_values"]
        calls.append((working.keys.copy(), working.conv.copy(), working.recurrent.copy()))
        working.keys[:] = -1
        return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])
    def pad(value, padding):
        pairs = list(zip(padding[::2], padding[1::2]))[::-1]
        return Array(np.pad(value, [(0, 0)] * (value.ndim - len(pairs)) + pairs))
    runtime = object.__new__(Decision3Runtime)
    runtime.torch = SimpleNamespace(
        tensor=Array, stack=lambda values, dim=0: Array(np.stack(values, axis=dim)),
        arange=lambda count, **kwargs: Array(np.arange(count)),
        cat=lambda values, dim=0: Array(np.concatenate(values, axis=dim)),
        inference_mode=nullcontext, nn=SimpleNamespace(functional=SimpleNamespace(pad=pad)),
    )
    runtime.native = SimpleNamespace(device="cpu", backbone=SimpleNamespace(language_model=language))
    runtime.helpers = SimpleNamespace(sdpa_backends=lambda _: nullcontext())
    runtime._readout = lambda hidden, rows: hidden
    tails = {(0, "a"): (Array([[1], [2]]), Array([[0, 1]])),
             (1, "b"): (Array([[3]]), Array([[0]])),
             (0, "c"): (Array([[4]]), Array([[0]]))}
    rows = [(0, "a"), (1, "b"), (0, "c")]
    state = (cache, Array([[True], [True]]), {0: 0, 1: 1}, tails)
    for _ in range(2):
        assert runtime._shared_logits(rows, state).tolist() == [[2], [3], [4]]
    assert cache.keys.tolist() == [[10], [20]]
    for keys, conv, recurrent in calls:
        assert keys.tolist() == [[10], [20], [10]]
        assert conv.tolist() == [[30], [40], [30]]
        assert recurrent.tolist() == [[50], [60], [50]]


@dataclass
class NativePrepared:
    keys: list
    questions: dict
    lengths: dict
    media: dict
    errors: dict = field(default_factory=dict)


class Scores:
    def __init__(self, values):
        self.values = values

    def __truediv__(self, _):
        return self

    def softmax(self, _):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


@pytest.mark.parametrize("fail_first", [False, True])
def test_batched_questions_keep_camera_identity_and_partial_results(fail_first):
    runtime = object.__new__(Decision3Runtime)
    runtime.shared_window_state = False
    runtime.torch = SimpleNamespace(cat=lambda values: sum(values, []))
    runtime._bind = lambda video, *_: video
    def respond(video, probabilities, tokens):
        return {"answers": {key: video.errors.get(key, probabilities.get(key)) for key in video.keys}}
    runtime.native = SimpleNamespace(
        temperature=1.0, synchronize=lambda: None, respond=respond,
        media_features=lambda media: {"video": [f"camera-{index}" for index in media["video_grid_thw"]]},
    )
    seen = []
    def logits(rows, prepared, features):
        seen.append([(index, key, features[index]) for index, key, *_ in rows])
        if fail_first and len(seen) == 1:
            raise ValueError("one question chunk failed")
        return Scores([[float(index), 1.0 - index, 0.0] for index, *_ in rows])
    runtime._logits = logits
    windows = []
    for index in range(2):
        keys = [f"q{i}" for i in range(6)]
        native = NativePrepared(keys, {key: SimpleNamespace(kind="choice") for key in keys},
                                dict.fromkeys(keys, 10), {"pixel_values_videos": [index], "video_grid_thw": [index]})
        video = _PreparedVideo(None, native, tuple((key, None, None, 3) for key in keys),
                               None, None, {"preprocess_seconds": 0.0})
        windows.append({"prepared": video, "question_template": None, "frame_times": [0.0, 0.5]})
    results = dict(runtime.predict_prepared_batch(windows))
    assert set(results) == {0, 1}
    assert [len(chunk) for chunk in seen] == [8, 4]
    assert all(feature == f"camera-{index}" for chunk in seen for index, _, feature in chunk)
    assert all(not window["prepared"].native.errors for window in windows)
    if fail_first:
        assert results[0][0]["answers"]["q0"]["error"] == "inference_failed"
        assert results[1][0]["answers"]["q0"]["error"] == "inference_failed"
    assert results[1][0]["answers"]["q5"] == [1.0, 0.0, 0.0]
