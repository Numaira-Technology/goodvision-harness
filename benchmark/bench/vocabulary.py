"""Pick the events each benchmark video's session should detect.

The harness detects nothing beyond the built-in five unless a session names
the events it wants. Declaring `events.definitions` in the harness config only
makes an event selectable; the runner still has to tick it per video.

So the runner reads the labels straight off the questions. A video asked about
`person_texts_on_phone` runs a session detecting that and nothing else, which
keeps the prompt short and stops the other twenty-five definitions from
competing for the model's attention on a video that no one asked about them.

The definitions are loaded from deployment config. Question-aware replay
selects registered labels from the public questions and options; other actions
remain in qa_observations. Legacy extraction callers can require an exact
vocabulary match with check_supported.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import yaml

from .schema import Question, question_target_label

VOCABULARY_PATH = Path(__file__).resolve().parents[1] / "events" / "vocabulary.yaml"


def load_definitions(path: str | Path | None = None) -> list[dict[str, Any]]:
    """The event definitions to declare in the harness config."""
    source = Path(path) if path is not None else VOCABULARY_PATH
    return yaml.safe_load(source.read_text(encoding="utf-8")) or []


def session_features(
    questions: Iterable[Question], *, supported: Iterable[str] | None = None,
) -> tuple[str, ...] | None:
    """Which events to enable for one video, or None to leave the default.

    With a supported vocabulary, select only registered labels from the public
    event names/options. The complete questions still reach qa_observations.
    None leaves the session on its default events, including when no queried
    action has a registered definition. Without supported, retain legacy label
    selection for capture/extraction callers.
    """
    if supported is not None:
        # Question-aware replay also observes actions with no scene-event
        # definition. Keep those questions intact in qa_observations; do not
        # register event_selection, answer combinations, or private GT as events.
        available = set(supported)
        labels = set()
        for question in questions:
            context = question.public_context
            if question.capability == "event_recognition" and question.answer_type == "single_choice":
                candidates = context.get("event_labels") or question.response_options or ()
            else:
                label = question_target_label(question)
                candidates = (label,) if label else ()
            for candidate in candidates:
                label = "_".join(str(candidate).split(":")[-1].lower().split())
                if label in available:
                    labels.add(label)
    else:
        labels = {
            label
            for question in questions
            if (label := question_target_label(question)) is not None
        }
    return tuple(sorted(labels)) or None


def required_labels(questions: Iterable[Question]) -> tuple[str, ...]:
    """Every event label this run needs the harness to know about."""
    labels = {
        label
        for question in questions
        if (label := question_target_label(question)) is not None
    }
    return tuple(sorted(labels))


def supported_labels(base_url: str, timeout: float = 30.0) -> frozenset[str]:
    """What the running harness will let a session select."""
    url = base_url.rstrip("/") + "/v1/ui/info"
    with urlopen(url, timeout=timeout) as response:
        info = json.loads(response.read().decode("utf-8"))
    return frozenset(str(item) for item in info.get("supported_features", ()))


def check_supported(required: Iterable[str], supported: Iterable[str]) -> None:
    """Fail before the first session rather than once per video."""
    missing = tuple(sorted(set(required) - set(supported)))
    if not missing:
        return
    raise RuntimeError(
        "the harness does not know these event labels: "
        + ", ".join(missing)
        + f". Declare them under `events.definitions` in the harness config and "
        f"restart it; `python -m bench.vocabulary` prints the block from "
        f"{VOCABULARY_PATH.name}."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the events.definitions block for the harness config."
    )
    parser.add_argument("--vocabulary", type=Path, default=None)
    args = parser.parse_args(argv)
    definitions = load_definitions(args.vocabulary)
    block = yaml.safe_dump(
        {"events": {"definitions": definitions}},
        allow_unicode=True,
        sort_keys=False,
        width=88,
    )
    sys.stdout.write(block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
