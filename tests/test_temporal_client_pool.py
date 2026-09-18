"""Service-owned VLM clients preserve connections without leaking run ownership."""

from __future__ import annotations

import gc
import weakref
from threading import Barrier, Event, Lock, Thread, get_ident

import pytest

from goodvision_harness.model_api import ModelAPIError
from goodvision_harness.temporal_client_pool import TemporalClientPool


class Inspector:
    def __init__(self, *, warmup_result=True, warmup_error=None):
        self.owner = get_ident()
        self.threads = [self.owner]
        self.jobs = []
        self.warmups = 0
        self.closes = 0
        self.closed = Event()
        self.started = Event()
        self.release = Event()
        self.release.set()
        self.warmup_result = warmup_result
        self.warmup_error = warmup_error

    def warmup(self):
        self.threads.append(get_ident())
        self.warmups += 1
        if self.warmup_error:
            raise self.warmup_error
        return self.warmup_result

    def inspect(self, job):
        self.threads.append(get_ident())
        self.jobs.append(job)
        self.started.set()
        assert self.release.wait(5), "test did not release its fake model request"
        return (job,)

    def close(self):
        self.threads.append(get_ident())
        self.closes += 1
        self.closed.set()


class Factory:
    def __init__(self, **options):
        self.options = options
        self.inspectors = []
        self.lock = Lock()

    def __call__(self):
        inspector = Inspector(**self.options)
        with self.lock:
            self.inspectors.append(inspector)
        return inspector


class Call:
    """Capture a blocking API call without hiding failures in worker threads."""

    def __init__(self, operation):
        self.result = None
        self.error = None
        self.done = Event()

        def run():
            try:
                self.result = operation()
            except BaseException as exc:
                self.error = exc
            finally:
                self.done.set()

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()

    def finish(self):
        assert self.done.wait(3), "pool operation did not finish"
        self.thread.join(timeout=1)
        assert not self.thread.is_alive()
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def pool():
    selected = TemporalClientPool()
    yield selected
    selected.close(timeout_seconds=2)


def test_same_configuration_reuses_client_and_owner_thread_across_runs(pool):
    factory = Factory()
    first, = pool.prepare("same-route-and-key", factory, 1)
    first_job, next_job = object(), object()
    assert first.inspect(first_job) == (first_job,)
    inspector, = factory.inspectors
    first.close()
    assert not inspector.closed.is_set()

    second, = pool.prepare("same-route-and-key", factory, 1)
    assert first is not second
    assert second.inspect(next_job) == (next_job,)
    first.close()  # A late cleanup from run 1 must not invalidate run 2.
    assert second.inspect(next_job) == (next_job,)
    assert factory.inspectors == [inspector]
    assert inspector.warmups == 1
    assert inspector.owner != get_ident()
    assert set(inspector.threads) == {inspector.owner}
    with pytest.raises(ModelAPIError):
        first.inspect(object())

    second.close()
    pool.close(timeout_seconds=2)
    assert inspector.closed.wait(1)
    assert inspector.closes == 1
    assert set(inspector.threads) == {inspector.owner}


@pytest.mark.parametrize("change", ["route", "credential", "parallelism"])
def test_configuration_change_retires_old_lease_and_closes_on_owner(pool, change):
    factory = Factory()
    old, = pool.prepare("route-A-key-1", factory, 1)
    old.inspect(object())
    previous = factory.inspectors[0]
    key = {"route": "route-C-key-1", "credential": "route-A-key-2"}.get(
        change, "route-A-key-1"
    )
    new = pool.prepare(key, factory, 2 if change == "parallelism" else 1)
    with pytest.raises(ModelAPIError):
        old.inspect(object())
    for lease in new:
        job = object()
        assert lease.inspect(job) == (job,)
    assert previous.closed.wait(1)
    assert previous.closes == 1
    assert set(previous.threads) == {previous.owner}
    old.close()
    assert new[0].inspect("new-run") == ("new-run",)


@pytest.mark.parametrize("warmup_error", [None, RuntimeError("warmup failed")])
def test_failed_warmup_does_not_prevent_inference(pool, warmup_error):
    factory = Factory(warmup_result=False, warmup_error=warmup_error)
    lease, = pool.prepare("configured", factory, 1)
    job = object()
    assert lease.inspect(job) == (job,)
    assert factory.inspectors[0].warmups == 1


def test_inspector_without_optional_warmup_is_supported(pool):
    class MinimalInspector:
        def inspect(self, job):
            return (job,)

        def close(self):
            pass

    lease, = pool.prepare("configured", MinimalInspector, 1)
    job = object()
    assert lease.inspect(job) == (job,)


def test_expired_acceptance_never_calls_underlying_inspection(pool):
    factory = Factory()
    accepted = Event()
    accepted.set()
    lease, = pool.prepare("configured", factory, 1, accept=lambda _: accepted.is_set())
    first = object()
    assert lease.inspect(first) == (first,)
    accepted.clear()
    with pytest.raises(ModelAPIError):
        lease.inspect(object())
    assert factory.inspectors[0].jobs == [first]


def test_retiring_busy_client_finishes_before_close_and_replacement(pool):
    factory = Factory()
    old, = pool.prepare("old-route", factory, 1)
    old.inspect("initialize")
    inspector = factory.inspectors[0]
    inspector.started.clear()
    inspector.release.clear()
    active = Call(lambda: old.inspect("in-flight"))
    try:
        assert inspector.started.wait(1)
        Call(pool.invalidate).finish()
        assert not inspector.closed.is_set()
        with pytest.raises(ModelAPIError):
            old.inspect("stale")
        new, = pool.prepare("new-route", factory, 1)
        pending = Call(lambda: new.inspect("new-run"))
        assert not inspector.closed.is_set()
        inspector.release.set()
        assert active.finish() == ("in-flight",)
        assert pending.finish() == ("new-run",)
        assert inspector.closed.wait(1)
        replacement = factory.inspectors[-1]
        assert replacement is not inspector
        assert replacement.owner == inspector.owner
        assert set(inspector.threads) == {inspector.owner}
    finally:
        inspector.release.set()


def test_shutdown_is_bounded_and_closes_busy_client_after_request_returns(pool):
    factory = Factory()
    lease, = pool.prepare("configured", factory, 1)
    lease.inspect("initialize")
    inspector = factory.inspectors[0]
    inspector.started.clear()
    inspector.release.clear()
    active = Call(lambda: lease.inspect("in-flight"))
    try:
        assert inspector.started.wait(1)
        Call(lambda: pool.close(timeout_seconds=0.01)).finish()
        assert not inspector.closed.is_set()
        with pytest.raises(ModelAPIError):
            lease.inspect("shutdown")
        inspector.release.set()
        assert active.finish() == ("in-flight",)
        assert inspector.closed.wait(1)
        assert inspector.closes == 1
        assert set(inspector.threads) == {inspector.owner}
    finally:
        inspector.release.set()


def test_actor_allows_only_one_active_and_one_pending_request(pool):
    factory = Factory()
    lease, = pool.prepare("configured", factory, 1)
    lease.inspect("initialize")
    inspector = factory.inspectors[0]
    inspector.started.clear()
    inspector.release.clear()
    active = Call(lambda: lease.inspect("active"))
    barrier, rejected = Barrier(4), Event()
    errors = []
    error_lock = Lock()

    def contend(job):
        barrier.wait(timeout=3)
        try:
            return lease.inspect(job)
        except ModelAPIError as exc:
            with error_lock:
                errors.append(exc)
                if len(errors) == 2:
                    rejected.set()
            return exc

    try:
        assert inspector.started.wait(1)
        contenders = [Call(lambda job=job: contend(job)) for job in range(3)]
        barrier.wait(timeout=3)
        assert rejected.wait(2), "extra callers were allowed to queue without a bound"
        assert len(errors) == 2
        assert all(error.code == "overloaded" for error in errors)
        inspector.release.set()
        assert active.finish() == ("active",)
        results = [call.finish() for call in contenders]
        assert sum(isinstance(value, ModelAPIError) for value in results) == 2
        assert len(inspector.jobs) == 3  # Initialize, active, and one pending.
    finally:
        inspector.release.set()


def test_four_actors_infer_concurrently_and_reuse_threads_across_revisions(pool):
    factory = Factory()
    leases = pool.prepare("first-revision", factory, 4)
    for lease in leases:
        lease.inspect("initialize")
    first = tuple(factory.inspectors)
    assert len(first) == 4
    assert len({inspector.owner for inspector in first}) == 4
    for inspector in first:
        inspector.started.clear()
        inspector.release.clear()
    calls = [Call(lambda lease=lease: lease.inspect(object())) for lease in leases]
    try:
        assert all(inspector.started.wait(1) for inspector in first)
    finally:
        for inspector in first:
            inspector.release.set()
    for call in calls:
        call.finish()
    next_leases = pool.prepare("second-revision", factory, 4)
    for lease in next_leases:
        lease.inspect("new-run")
    second = factory.inspectors[4:]
    assert len(second) == 4
    assert {inspector.owner for inspector in first} == {
        inspector.owner for inspector in second
    }
    assert all(inspector.closed.wait(1) for inspector in first)
    assert all(set(inspector.threads) == {inspector.owner} for inspector in first)


def test_close_cancels_pending_lease_without_closing_shared_client(pool):
    factory = Factory()
    old, = pool.prepare("same-route", factory, 1)
    old.inspect("initialize")
    inspector = factory.inspectors[0]
    inspector.started.clear()
    inspector.release.clear()
    active = Call(lambda: old.inspect("active"))
    barrier, rejected = Barrier(4), Event()
    errors = []
    error_lock = Lock()

    def contend(job):
        barrier.wait(timeout=3)
        try:
            return old.inspect(job)
        except ModelAPIError as exc:
            with error_lock:
                errors.append(exc)
                if len(errors) >= 2:
                    rejected.set()
            return exc

    try:
        assert inspector.started.wait(1)
        contenders = [Call(lambda job=job: contend(job)) for job in range(3)]
        barrier.wait(timeout=3)
        assert rejected.wait(2)  # One pending job remains behind the active call.
        assert len(errors) == 2
        Call(old.close).finish()
        assert all(isinstance(call.finish(), ModelAPIError) for call in contenders)
        assert inspector.jobs == ["initialize", "active"]
        assert not inspector.closed.is_set()
        inspector.release.set()
        with pytest.raises(ModelAPIError):
            active.finish()  # Closing a run also prevents its late result leaking out.
        new, = pool.prepare("same-route", factory, 1)
        assert new.inspect("next-run") == ("next-run",)
        assert factory.inspectors == [inspector]
        assert not inspector.closed.is_set()
    finally:
        inspector.release.set()


def test_acceptance_is_checked_again_after_response_arrives(pool):
    factory = Factory()
    accepted = Event()
    accepted.set()
    lease, = pool.prepare("configured", factory, 1, accept=lambda _: accepted.is_set())
    lease.inspect("initialize")
    inspector = factory.inspectors[0]
    inspector.started.clear()
    inspector.release.clear()
    active = Call(lambda: lease.inspect("in-flight"))
    try:
        assert inspector.started.wait(1)
        accepted.clear()
        inspector.release.set()
        with pytest.raises(ModelAPIError):
            active.finish()
        assert inspector.jobs == ["initialize", "in-flight"]
    finally:
        inspector.release.set()


def test_pool_cannot_be_reopened_after_service_shutdown(pool):
    pool.close(timeout_seconds=0)
    with pytest.raises(ModelAPIError):
        pool.prepare("new-run", Factory(), 1)


@pytest.mark.parametrize("size", [0, 5, True, 1.5])
def test_actor_count_is_bounded(pool, size):
    with pytest.raises(ValueError):
        pool.prepare("configured", Factory(), size)


def test_idle_actor_releases_previous_job_result_and_run_predicate(pool):
    class RunOwner:
        def accept(self, _job):
            return True

    class Job:
        pass

    class Response:
        def __init__(self, job):
            self.job = job

    class EphemeralInspector:
        def inspect(self, job):
            return Response(job)

        def close(self):
            pass

    owner, job = RunOwner(), Job()
    lease, = pool.prepare("configured", EphemeralInspector, 1, accept=owner.accept)
    response = lease.inspect(job)
    released = [Event() for _ in range(4)]
    references = [
        weakref.ref(value, lambda _ref, event=event: event.set())
        for value, event in zip((job, response, owner, lease), released, strict=True)
    ]
    lease.close()
    del job, response, owner, lease
    gc.collect()
    # Future completion can wake the caller just before the actor's finally.
    # Weakref callbacks observe that cleanup without a timing-dependent sleep.
    assert all(event.wait(2) for event in released)
    assert all(reference() is None for reference in references)


def test_invalidating_pool_releases_closed_client_and_bound_warmup(pool):
    closed, released = Event(), Event()
    references = []

    class CollectibleInspector:
        def warmup(self):
            return True

        def inspect(self, _job):
            return ()

        def close(self):
            closed.set()

    def factory():
        inspector = CollectibleInspector()
        references.append(weakref.ref(inspector, lambda _ref: released.set()))
        return inspector

    lease, = pool.prepare("configured", factory, 1)
    assert lease.inspect(object()) == ()
    pool.invalidate()
    assert closed.wait(2)
    gc.collect()
    assert released.wait(2), "idle actor retained its retired inspector"
    assert len(references) == 1
    assert references[0]() is None
