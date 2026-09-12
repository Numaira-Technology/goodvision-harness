"""Bounded native EOF postprocessing contracts, without GPU execution."""

from types import SimpleNamespace

import pytest

from workers.sam_scene.continuous import SessionSettings
from workers.sam_scene.streaming_state import finalize_pending, make_prod_model


def fixture(count, batch_size=4):
    calls = []
    state = {
        "num_frames": 20,
        "orig_height": 40,
        "orig_width": 64,
        "generator_state": {
            "hotstart_buffer": [
                (index, {"i": index, "suppressed_obj_ids": {index + 100}})
                for index in range(20 - count, 20)
            ],
            "postprocess_yield_list": [],
            "hotstart_removed_obj_ids": {500},
            "unconfirmed_obj_ids_per_frame": {index: {index} for index in range(20)},
        },
    }

    def postprocess(height, width, inputs):
        assert (height, width) == (40, 64)
        assert 1 <= len(inputs) <= batch_size
        calls.append(inputs)
        return [{"frame": item[0]["i"], "objects": []} for item in inputs]

    model = SimpleNamespace(
        world_size=1,
        postprocess_batch_size=batch_size,
        masklet_confirmation_consecutive_det_thresh=3,
        _postprocess_output_batched=postprocess,
        _postprocess_output=lambda state, *args: postprocess(40, 64, [args])[0],
    )
    return model, state, calls


@pytest.mark.parametrize(
    "count,groups",
    [(0, []), (1, [1]), (2, [2]), (4, [4]), (5, [4, 1]), (14, [4, 4, 4, 2])],
)
def test_empty_eof_caps_each_native_call_and_preserves_empty_person_frames(
    count, groups
):
    model, state, calls = fixture(count)
    outputs = list(finalize_pending(model, state))
    assert [len(call) for call in calls] == groups
    assert [index for index, _ in outputs] == list(range(20 - count, 20))
    assert all(
        output["frame"] == index and output["objects"] == []
        for index, output in outputs
    )
    for inputs in calls:
        for output, removed, suppressed, unconfirmed in inputs:
            index = output["i"]
            assert removed == {500}
            assert suppressed == {index + 100}
            assert unconfirmed == {min(index + 2, 19)}
    assert state["generator_state"]["hotstart_buffer"] == []
    assert state["num_frames"] == 20


def test_four_frame_and_single_frame_eof_have_same_source_and_filter_metadata():
    one, state_one, calls_one = fixture(14, batch_size=1)
    four, state_four, calls_four = fixture(14, batch_size=4)
    assert list(finalize_pending(one, state_one)) == list(
        finalize_pending(four, state_four)
    )
    assert [item for call in calls_one for item in call] == [
        item for call in calls_four for item in call
    ]


def test_short_native_result_does_not_clear_pending_or_silently_drop_source():
    model, state, _ = fixture(5)
    model._postprocess_output_batched = lambda *_: []
    with pytest.raises(RuntimeError, match="omitted source frames"):
        list(finalize_pending(model, state))
    assert len(state["generator_state"]["hotstart_buffer"]) == 5


def test_cancelled_generator_does_not_clear_uncommitted_tail():
    model, state, _ = fixture(14)
    pending = finalize_pending(model, state)
    assert next(pending)[0] == 6
    pending.close()
    # Caller invalidates this epoch; the helper never falsely marks it drained.
    assert len(state["generator_state"]["hotstart_buffer"]) == 14


@pytest.mark.parametrize("value", [True, 0, 2, 3, 5, 16, "4", 4.0])
def test_unsupported_postprocessing_sizes_fail_before_model_import(value):
    with pytest.raises(ValueError, match="postprocess_batch_size"):
        SessionSettings(postprocess_batch_size=value)
    with pytest.raises(ValueError, match="postprocess_batch_size"):
        make_prod_model(None, postprocess_batch_size=value)


def test_postprocessing_default_and_environment_are_explicit(monkeypatch):
    assert SessionSettings().postprocess_batch_size == 1
    monkeypatch.setenv("SAM31_SESSION_POSTPROCESS_BATCH_SIZE", "4")
    assert SessionSettings.from_env().postprocess_batch_size == 4


def test_unflushed_previous_postprocessing_state_is_not_accepted():
    model, state, _ = fixture(4)
    state["generator_state"]["postprocess_yield_list"] = [object()]
    with pytest.raises(RuntimeError, match="unflushed"):
        list(finalize_pending(model, state))
