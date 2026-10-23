"""Compatibility exports for the packaged VideoJev question contract."""

from goodvision_harness.videojev_contract import (
    CONTEXT,
    LABELS,
    MAX_PEOPLE,
    MODEL,
    OPTIONS,
    REVISION,
    VALUES,
    adapt,
    experiment,
    parse_choices,
    questions,
    values_for_question,
)

__all__ = (
    "CONTEXT", "LABELS", "MAX_PEOPLE", "MODEL", "OPTIONS", "REVISION", "VALUES",
    "adapt", "experiment", "parse_choices", "questions", "values_for_question",
)
