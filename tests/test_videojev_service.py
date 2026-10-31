"""CPU-only startup checks; no native models, CUDA, downloads or sockets."""

import base64
import io
import json
import sys
from threading import Event
from types import SimpleNamespace

import pytest
from PIL import Image

from goodvision_harness import videojev_service as service


@pytest.mark.parametrize(
    "sizes,cap,selected,remaining",
    [
        ([10, 100, 11], 2, [0, 2], [1]),
        ([10, 11, 9], 2, [0, 1], [2]),
        ([10, 100, 11, 12, 10], 3, [0, 2, 3], [1, 4]),
        ([10, 11], 4, [0, 1], []),
        ([10, 11], 1, [0], [1]),
    ],
)
def test_ready_selection_minimizes_padding_and_preserves_order(sizes, cap, selected, remaining):
    from collections import deque

    template = object()
    works = [SimpleNamespace(
        index=index, state="ready", question_template=template,
        prepared=SimpleNamespace(record={"tokens": size}),
    ) for index, size in enumerate(sizes)]
    scorer = object.__new__(service.Scorer)
    scorer.runtime = SimpleNamespace(batch_enabled=True, predict_prepared_batch=lambda _: None)
    scorer.gpu_batch_size = cap
    scorer._pipeline = deque(works)
    assert [work.index for work in scorer._take_ready()] == selected
    assert [work.index for work in scorer._pipeline] == remaining


@pytest.mark.parametrize("barrier", ["preparing", "template", "cache_disabled"])
def test_ready_selection_does_not_cross_incompatible_work(barrier):
    from collections import deque

    template = object()
    works = [SimpleNamespace(
        index=index, state="ready", question_template=template,
        prepared=SimpleNamespace(record={"tokens": 10}),
    ) for index in range(3)]
    scorer = object.__new__(service.Scorer)
    scorer.runtime = SimpleNamespace(batch_enabled=barrier != "cache_disabled", predict_prepared_batch=lambda _: None)
    scorer.gpu_batch_size = 2
    scorer._pipeline = deque(works)
    if barrier == "preparing":
        works[1].state = "preparing"
    elif barrier == "template":
        works[1].question_template = object()
    assert scorer._take_ready() == works[:1]
    assert list(scorer._pipeline) == works[1:]


@pytest.mark.parametrize("cap,cpu_threads", [(1, None), (2, 1), (4, 8)])
def test_startup_warmup_respects_gpu_batch_cap_before_serving(
    tmp_path, monkeypatch, cap, cpu_threads
):
    calls, prepared, runtimes, events = [], [], [], []
    thread_settings = []

    class Runtime:
        def __init__(self, *args, **kwargs):
            self.metadata = {}
            self.prefix_cache = True
            runtimes.append(self)

        def prepare(self, payloads, fps, **kwargs):
            with Image.open(io.BytesIO(payloads[0])) as image:
                size = image.size
            prepared.append(size)
            return size

        def predict_prepared(self, prepared, **kwargs):
            calls.append(("single", (prepared,)))
            return [], {}

        def predict_prepared_batch(self, windows):
            assert len(windows) <= cap, "startup exceeded configured GPU batch cap"
            calls.append(("batch", tuple(w["prepared"] for w in windows)))
            for index in range(len(windows)):
                yield index, ([], {})

    class Scorer:
        def __init__(self, factory, *args, **kwargs):
            assert kwargs["gpu_batch_size"] == cap
            self.runtime = factory()
            events.append("warm")

        def wait_ready(self):
            events.append("ready")

        def snapshot(self):
            return {"ready": True}

        def shutdown(self, **kwargs):
            pass

    class Server:
        def __init__(self, *args):
            assert events == ["warm", "ready"]
            events.append("listen")

        def serve_forever(self):
            pass

        def server_close(self):
            pass

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda value: thread_settings.append(("cpu", value)),
        set_num_interop_threads=lambda value: thread_settings.append(("interop", value)),
    ))
    monkeypatch.setattr(service, "VideoJevRuntime", Runtime)
    monkeypatch.setattr(service, "Scorer", Scorer)
    monkeypatch.setattr(service, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(sys, "argv", [
        "videojev-service", "--cache-dir", str(tmp_path),
        "--result-log", str(tmp_path / "results.jsonl"),
        "--gpu-batch-size", str(cap),
    ] + ([] if cpu_threads is None else ["--cpu-threads", str(cpu_threads)]))
    service.main()

    assert thread_settings == [("cpu", cpu_threads or 4), ("interop", 4)]
    assert prepared == [(640, 360), (720, 405)]
    assert {size for _kind, windows in calls for size in windows} == set(prepared)
    assert all(len(windows) <= cap for _kind, windows in calls)
    assert [kind for kind, _windows in calls] == (
        ["single", "single"] if cap == 1 else ["single", "batch"]
    )
    assert runtimes[0].startup_template.rows
    assert runtimes[0].metadata["startup_prepare_seconds"] >= 0
    assert events == ["warm", "ready", "listen"]


@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_cpu_threads_fail_before_runtime_initialization(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", [
        "videojev-service", "--cache-dir", "unused", "--result-log", "unused",
        "--cpu-threads", value,
    ])
    with pytest.raises(SystemExit) as error:
        service.main()
    assert error.value.code == 2


@pytest.mark.parametrize("cap,ready_capacity", [(1, 4), (3, 4), (4, 4), (4, 2)])
def test_scorer_batches_ready_windows_within_configured_cap(
    tmp_path, cap, ready_capacity
):
    entered, release = Event(), Event()
    calls = []
    rows = [{
        "id": "scene/fight", "task": "choice",
        "query_parts": ["Source times: ", ". Evaluate fight."],
        "options": ["Present", "Absent", "Uncertain"],
    }]

    def predictions(prepared, template):
        selected = prepared.value % 3
        return [{
            "id": row["id"], "task": "choice", "answer_index": selected,
            "answer": row["options"][selected],
            "probabilities": [float(index == selected) for index in range(3)],
        } for row in template.rows]

    class Runtime:
        prefix_cache = True

        def __init__(self):
            self.metadata = {}

        def prepare(self, images, fps, **kwargs):
            return SimpleNamespace(value=int(images[0]), record={"tokens": 16})

        def predict_prepared(self, prepared, *, question_template, frame_times):
            calls.append(("single", (prepared.value,)))
            if prepared.value == 0:
                entered.set()
                assert release.wait(5)
            return predictions(prepared, question_template), {}

        def predict_prepared_batch(self, windows):
            assert len({id(w["question_template"]) for w in windows}) == 1
            calls.append(("batch", tuple(w["prepared"].value for w in windows)))
            for index, window in enumerate(windows):
                yield index, (
                    predictions(window["prepared"], window["question_template"]), {},
                )

    def body(index):
        return {
            "request_id": str(index), "session": f"camera-{index}", "fps": 2,
            "frame_times": [0.0, 0.5],
            "images": [base64.b64encode(str(index).encode()).decode()] * 2,
            "question_template": rows,
        }

    audit = tmp_path / "results.jsonl"
    scorer = service.Scorer(
        Runtime, audit, gpu_batch_size=cap, ready_capacity=ready_capacity,
    )
    try:
        scorer.wait_ready()
        futures = [scorer.submit(body(0))]
        assert entered.wait(5)
        futures.extend(scorer.submit(body(index)) for index in range(1, 5))
        # Occupy the GPU with the first source while CPU preparation fills the
        # independently configured ready capacity. The next dispatch is then
        # deterministic and must neither wait to fill nor exceed the GPU cap.
        with scorer._condition:
            assert scorer._condition.wait_for(
                lambda: len(scorer._pipeline) == ready_capacity
                and all(work.state == "ready" for work in scorer._pipeline),
                timeout=5,
            )
        release.set()
        results = [future.result(timeout=5) for future in futures]
        assert [result["request_id"] for result in results] == list(map(str, range(5)))
        assert [result["raw"][0]["answer_index"] for result in results] == [
            index % 3 for index in range(5)
        ]
        assert sorted(index for _kind, indices in calls for index in indices) == [
            0, 1, 2, 3, 4,
        ]
        assert all(len(indices) <= cap for _kind, indices in calls)
        assert len(calls[1][1]) == min(cap, ready_capacity)
        assert calls[1][0] == ("single" if cap == 1 else "batch")
        assert scorer.healthy
    finally:
        release.set()
        scorer.shutdown()
    saved = [
        json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()
    ]
    assert len(saved) == 5
    assert {row["request_id"] for row in saved} == set(map(str, range(5)))
    assert {row["status"] for row in saved} == {"completed"}
