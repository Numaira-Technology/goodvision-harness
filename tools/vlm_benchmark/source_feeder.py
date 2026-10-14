"""Keep scheduled video input independent of report/UI collection work."""

import time
from queue import Empty, Queue
from threading import Event, Thread


class TimedSourceFeeder:
    def __init__(
        self, branches, framesets, releases, started, sample_fps,
        *, monotonic=None, wait=None,
    ):
        self.branches = branches
        self.framesets = framesets
        self.releases = releases
        self.started = started
        self.sample_fps = sample_fps
        # At most one small notification per planned source sample. Frames and
        # model results stay in their existing bounded production transports.
        self.samples = Queue(maxsize=max(1, sum(len(a) for _, a in releases)))
        self.stop = Event()
        # The stage start and release lag must share one monotonic clock.
        # Production waits remain interruptible; offline schedulers can supply
        # a matching virtual wait without touching host clock functions.
        self._monotonic = monotonic or time.monotonic
        self._wait = wait or self.stop.wait
        self.error = None
        self.release_lags = []
        self.missed = 0
        self.missed_by_stream = [0] * len(branches)
        self.offer_rejected = [0] * len(branches)
        self.thread = Thread(
            target=self._run, name="benchmark-source-feeder", daemon=True
        )

    def _run(self):
        try:
            for release_at, actions in self.releases:
                due = self.started + release_at
                if self.stop.is_set() or self._wait(
                    max(0.0, due - self._monotonic())
                ):
                    return
                for camera, tick in actions:
                    if self.stop.is_set():
                        return
                    # Synchronized sources share a release slot, not an actual
                    # offer time. One slow camera must not hide late siblings.
                    lag = self._monotonic() - due
                    self.release_lags.append(lag)
                    if lag > 1 / self.sample_fps:
                        self.missed += 1
                        self.missed_by_stream[camera] += 1
                        continue
                    frames = self.framesets[camera % len(self.framesets)]
                    accepted = self.branches[camera].offer(
                        frames[tick % len(frames)], tick / self.sample_fps, tick
                    )
                    self.offer_rejected[camera] += int(not accepted)
                    self.samples.put_nowait((camera, tick, accepted))
        except BaseException as exc:
            self.error = exc

    def start(self):
        self.thread.start()

    def notify(self, observer):
        # The UI/logging stays on the collector thread; it cannot delay offers.
        while True:
            try:
                sample = self.samples.get_nowait()
            except Empty:
                return
            if observer is not None:
                observer.sample(*sample)

    def check(self):
        if self.error is not None:
            raise self.error

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("source feeder did not stop")
