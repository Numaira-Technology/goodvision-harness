"""Shared fixtures for the preserved experimental runtime, separate from main."""

import time
from dataclasses import fields

from test_vlm_independent import BOX, ROOT, await_condition, job, jpeg
from test_vlm_independent import config as main_config

from tools.benchmark_runtime.vlm_independent import (
    SourceSample,
    VLMProcessConfig,
    utc_now,
)
from tools.benchmark_runtime.vlm_independent_model import IndependentVerdict

__all__ = [
    "BOX",
    "ROOT",
    "await_condition",
    "config",
    "job",
    "jpeg",
    "report",
    "utc_now",
    "verdict",
]


def config(session="vlm-run", generation=1):
    base = main_config(session, generation)
    return VLMProcessConfig(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    )


def verdict(detected=True, sid="p1", box=BOX):
    return IndependentVerdict("smoking", (sid,), detected, 0.9, ((sid, box),))


def report(state, at, sequence, verdicts):
    inspection = job(state.config, at, sequence)
    sample = SourceSample(inspection.frames[-1], utc_now(), time.monotonic())
    return state.report(inspection, sample, verdicts, {"model_seconds": 0.1})
