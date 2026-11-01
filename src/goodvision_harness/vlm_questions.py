"""Bind fixed choice templates once per runtime, before GPU inference."""

from dataclasses import dataclass, field
from threading import Lock


@dataclass(eq=False)
class QuestionTemplate:
    """Share compilation ownership while keeping native tokenization model-specific.

    A runtime implements compile_questions(rows); its compiled object implements
    bind(...). CPU preparation saves that result with the window. GPU prediction
    consumes it without rebuilding questions or sharing video state across cameras.
    """

    rows: tuple
    compiled: object = None
    _runtime: object = field(default=None, init=False, repr=False)
    _lock: object = field(default_factory=Lock, init=False, repr=False)

    def bind(self, runtime, *args, **kwargs):
        with self._lock:
            if self.compiled is None or self._runtime is not runtime:
                compiled = runtime.compile_questions(self.rows)
                self.compiled, self._runtime = compiled, runtime
            return self.compiled.bind(*args, **kwargs)
