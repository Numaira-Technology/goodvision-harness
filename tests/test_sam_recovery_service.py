"""Recovery callbacks cannot replace video identity or revive obsolete SAM runs.

All inputs are in-memory doubles; no camera, HTTP request or inference runs.
"""

from threading import RLock
from types import SimpleNamespace

import pytest
from test_camera_display_protocol import RUN, batch, header, source

from goodvision_harness import service
from goodvision_harness.camera_display import CameraDisplayMetadata


def setup_runtime():
    # Exercise the real service boundary without starting unrelated workers.
    runtime = service.HarnessRuntime.__new__(service.HarnessRuntime)
    runtime._state_lock = RLock()
    runtime._source_commit_lock = RLock()
    runtime._closed = False
    runtime._rich_config_revision = 7
    runtime._active_rich_session_id = RUN
    runtime._session = SimpleNamespace(stream_session_id=RUN)
    coordinator = SimpleNamespace(
        _generation=3,
        _closed=False,
        on_native_batch=None,
        on_native_recovery=None,
    )
    runtime._rich_labels = coordinator
    runtime._camera_display_metadata = CameraDisplayMetadata()
    runtime._camera_display_metadata.reset(
        RUN, native_token=(7, id(coordinator), 2),
    )
    assert runtime._camera_display_metadata.accept_source(header(10, 1.0))
    return runtime, coordinator


def test_recovery_rebind_accepts_new_native_token_and_preserves_video_source():
    runtime, coordinator = setup_runtime()
    metadata = runtime._camera_display_metadata
    identity, last_source = metadata.source_identity(), metadata.last_source
    old_token, new_token = (7, id(coordinator), 2), (7, id(coordinator), 3)

    assert runtime._bind_recovered_native_run(7, coordinator, RUN, 3)
    assert metadata.source_identity() == identity
    assert metadata.last_source == last_source
    assert not metadata.observe_batch(batch(source(11, 1.1)), native_token=old_token)
    assert metadata.observe_batch(batch(source(11, 1.1)), native_token=new_token)
    # Tracking recovery does not require a new video run or reset its clock.
    assert metadata.accept_source(header(12, 1.2))
    assert metadata.source_identity() == identity
    assert metadata.last_source == (12, 1.2)


@pytest.mark.parametrize(
    "stale",
    ["revision", "coordinator", "stream", "generation", "runtime_closed",
     "coordinator_closed", "no_session", "inactive_stream"],
)
def test_obsolete_recovery_callback_does_not_rebind_native_display(stale):
    runtime, coordinator = setup_runtime()
    revision, candidate, stream, generation = 7, coordinator, RUN, 3
    if stale == "revision":
        revision -= 1
    elif stale == "coordinator":
        candidate = SimpleNamespace(_generation=3, _closed=False)
    elif stale == "stream":
        stream = "retired-camera-run"
    elif stale == "generation":
        generation -= 1
    elif stale == "runtime_closed":
        runtime._closed = True
    elif stale == "coordinator_closed":
        coordinator._closed = True
    elif stale == "no_session":
        runtime._session = None
    else:
        runtime._active_rich_session_id = "replacement-camera-run"

    metadata = runtime._camera_display_metadata
    identity = metadata.source_identity()
    assert not runtime._bind_recovered_native_run(
        revision, candidate, stream, generation,
    )
    assert metadata.source_identity() == identity
    assert metadata.last_source == (10, 1.0)
    assert metadata.observe_batch(
        batch(source(11, 1.1)), native_token=(7, id(coordinator), 2),
    )
    assert not metadata.observe_batch(
        batch(source(12, 1.2)), native_token=(7, id(coordinator), 3),
    )


def test_tracking_coordinator_wires_recovery_to_service_epoch_guard(monkeypatch):
    runtime, coordinator = setup_runtime()
    rich = SimpleNamespace(backend="sam31_stream", processing_method="sam_live")
    monkeypatch.setattr(
        service, "create_tracking_coordinator",
        lambda client, rich, on_batch: coordinator,
    )
    called = []
    monkeypatch.setattr(
        runtime, "_bind_recovered_native_run",
        lambda *args: called.append(args) or True,
    )
    assert runtime._tracking_coordinator(object(), rich) is coordinator
    assert coordinator.on_native_recovery(RUN, 4)
    assert called == [(7, coordinator, RUN, 4)]


@pytest.mark.parametrize("invalidated", ["before_read", "during_read", None])
def test_retired_epoch_cannot_commit_across_retained_source_read(invalidated):
    runtime, coordinator = setup_runtime()
    native_batch = batch(source(11, 1.1))
    accepted = invalidated != "before_read"
    loaded, committed, checked = [], [], []

    def accepts(candidate):
        checked.append(candidate)
        return accepted

    def retained_sources(candidate):
        nonlocal accepted
        loaded.append(candidate)
        if invalidated == "during_read":
            accepted = False
        return ("retained-original-frame",)

    coordinator.accepts_batch = accepts
    coordinator.retained_sources = retained_sources
    runtime._on_tracking_batch = lambda candidate, **kwargs: committed.append(
        (candidate, kwargs),
    )
    runtime._commit_processing_batch(7, native_batch)

    assert checked
    if invalidated == "before_read":
        assert loaded == committed == []
    elif invalidated == "during_read":
        assert loaded == [native_batch]
        assert len(checked) >= 2
        assert committed == []
    else:
        assert loaded == [native_batch]
        assert committed == [
            (native_batch, {"retained_sources": ("retained-original-frame",)}),
        ]
