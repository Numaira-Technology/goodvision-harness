"""Hybrid-cache isolation and bounded reuse without Torch, CUDA or model weights."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from goodvision_harness import videojev_runtime
from goodvision_harness.videojev_questions import _Prefix


class Tensor:
    """Model only the tensor storage operations used when restoring a cache."""

    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    @property
    def device(self):
        return "cpu"

    def expand(self, *shape):
        return Tensor(np.broadcast_to(self.values, shape))

    def copy_(self, source):
        np.copyto(self.values, source.values)


class HybridCache:
    def __init__(self):
        self.rows = 1
        self.layers = [
            SimpleNamespace(
                keys=Tensor(np.arange(6).reshape(1, 1, 3, 2)),
                values=Tensor(np.arange(6, 12).reshape(1, 1, 3, 2)),
            ),
            SimpleNamespace(
                conv_states={0: Tensor(np.arange(8).reshape(1, 2, 4)), 1: None},
                recurrent_states={
                    0: Tensor(np.arange(18).reshape(1, 2, 3, 3)), 1: None,
                },
            ),
        ]

    def reorder_cache(self, indices):
        self.rows = len(indices)
        attention, linear = self.layers
        for name in ("keys", "values"):
            original = getattr(attention, name)
            setattr(attention, name, Tensor(original.values[indices].copy()))
        for name in ("conv_states", "recurrent_states"):
            states = getattr(linear, name)
            for index, value in states.items():
                if value is not None:
                    states[index] = Tensor(value.values[indices].copy())


def runtime_and_prefix():
    runtime = object.__new__(videojev_runtime.VideoJevRuntime)
    runtime.torch = SimpleNamespace(
        zeros=lambda count, **kwargs: np.zeros(count, dtype=np.int64),
        long=np.int64,
        bfloat16="bfloat16",
        autocast=lambda *args, **kwargs: nullcontext(),
        cat=lambda values, dim=0: Tensor(np.concatenate([
            value.values if isinstance(value, Tensor) else value
            for value in values
        ], axis=dim)),
    )
    return runtime, _Prefix(ids=None, tokens=(1,), state=HybridCache())


def assert_restored(working, original, count):
    for actual, expected in zip(working.layers, original.layers, strict=True):
        for name in ("keys", "values"):
            if hasattr(expected, name):
                source = getattr(expected, name).values
                np.testing.assert_array_equal(
                    getattr(actual, name).values,
                    np.broadcast_to(source, (count, *source.shape[1:])),
                )
        for name in ("conv_states", "recurrent_states"):
            for index, source in getattr(expected, name, {}).items():
                target = getattr(actual, name)[index]
                if source is None:
                    assert target is None
                else:
                    np.testing.assert_array_equal(
                        target.values,
                        np.broadcast_to(source.values, (count, *source.shape[1:])),
                    )


@pytest.mark.parametrize("count", [1, 2, 4])
def test_working_cache_restores_independent_rows_after_each_window(count):
    runtime, prefix = runtime_and_prefix()
    baseline = deepcopy(prefix.state)
    working = runtime._working_prefix(prefix, count, "cpu")
    assert_restored(working, baseline, count)
    linear = working.layers[1]
    buffers = [linear.conv_states[0], linear.recurrent_states[0]]

    # A model forward replaces dynamic attention KV and mutates linear buffers,
    # including before an inference exception. Neither may contaminate a later
    # window or the immutable prefix used by a different batch cardinality.
    attention = working.layers[0]
    attention.keys = Tensor(np.full((count, 1, 5, 2), 999))
    attention.values = Tensor(np.full((count, 1, 5, 2), 999))
    for buffer in buffers:
        buffer.values[0] += 1000
        if count > 1:
            assert (buffer.values[1] < 1000).all()
    restored = runtime._working_prefix(prefix, count, "cpu")
    assert restored is working
    assert restored.layers[1].conv_states[0] is buffers[0]
    assert restored.layers[1].recurrent_states[0] is buffers[1]
    assert_restored(restored, baseline, count)
    assert_restored(prefix.state, baseline, 1)

    other_count = 2 if count == 1 else 1
    other = runtime._working_prefix(prefix, other_count, "cpu")
    assert_restored(other, baseline, other_count)
    other.layers[1].conv_states[0].values[:] += 500
    assert_restored(restored, baseline, count)
    assert_restored(prefix.state, baseline, 1)


@pytest.mark.parametrize("count", [1, 2, 4])
def test_failed_language_forward_cannot_contaminate_next_window(count):
    runtime, prefix = runtime_and_prefix()
    baseline = deepcopy(prefix.state)
    working = runtime._working_prefix(prefix, count, "cpu")

    def failed_language(**kwargs):
        active = kwargs["past_key_values"]
        assert active is working
        active.layers[0].keys = Tensor(np.full((count, 1, 5, 2), 999))
        active.layers[1].conv_states[0].values[:] += 1000
        active.layers[1].recurrent_states[0].values[:] += 1000
        raise RuntimeError("partial model failure")

    runtime.model = SimpleNamespace(
        backbone=SimpleNamespace(language_model=failed_language),
    )
    runtime._prefix_inputs = lambda *args: (
        Tensor(np.ones((1, 2, 4))), np.zeros((3, 1, 3)), {},
    )
    with pytest.raises(RuntimeError, match="partial model failure"):
        runtime._score_cached([{}] * count, [None] * count, prefix)
    assert (working.layers[1].conv_states[0].values >= 1000).all()
    assert_restored(prefix.state, baseline, 1)
    assert runtime._working_prefix(prefix, count, "cpu") is working
    assert_restored(working, baseline, count)


def test_varied_batch_sizes_bound_retained_rows_and_evict_before_allocation(
    monkeypatch,
):
    runtime, prefix = runtime_and_prefix()
    retained_before_allocation = []

    def observed_copy(value):
        retained_before_allocation.append(len(prefix.working_states))
        return deepcopy(value)

    monkeypatch.setattr(videojev_runtime, "deepcopy", observed_copy)
    for count in range(1, 17):
        runtime._working_prefix(prefix, count, "cpu")
        assert len(prefix.working_states) <= 2
        assert sum(cache.rows for cache in prefix.working_states.values()) <= 2 * count
    assert retained_before_allocation == [0, *([1] * 15)]
    assert list(prefix.working_states) == [15, 16]
    recent = prefix.working_states[15]
    assert runtime._working_prefix(prefix, 15, "cpu") is recent
    assert list(prefix.working_states) == [16, 15]
    runtime._working_prefix(prefix, 17, "cpu")
    assert list(prefix.working_states) == [15, 17]
    assert sum(cache.rows for cache in prefix.working_states.values()) == 32
