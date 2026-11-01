"""Compatibility entry point for the isolated historical benchmark inspector."""

from dataclasses import replace

from tools.benchmark_runtime.person_grounding import (
    TrackedPersonInspector as TrackedPersonInspector,
)
from tools.benchmark_runtime.person_grounding import (
    TrackedVerdict as TrackedVerdict,
)
from tools.benchmark_runtime.person_grounding import (
    parse_verdicts as parse_verdicts,
)
from tools.vlm_benchmark.modes import mode_inspector


def inspector(config, *, broker_url):
    return mode_inspector(
        replace(
            config,
            grounding_source="detector_ids",
            detector_url=None,
            detector_broker_url=broker_url,
        )
    )
