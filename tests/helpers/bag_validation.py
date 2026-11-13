"""Synthetic bag references shared by offline benchmark regression tests."""

from tools.unified_benchmark.contract import BAG_VALIDATION_SCHEMA, EVENTS, digest


def bag_manifest():
    """Ten original clips, two questions each; never real calibration evidence."""
    return {
        "schema": BAG_VALIDATION_SCHEMA,
        "videos": [
            {
                "filename": f"supplement-{i}.mp4",
                "sha256": digest(f"supplement-{i}"),
                # Recording cuts can share a source group within validation.
                "source_video_group": f"supplement-source-{i // 2}",
                "questions": [
                    {
                        "event": event,
                        "proposed_answer": "yes" if i % 2 else "no",
                        "human_review": "synthetic label, not human reviewed",
                    }
                    for event in EVENTS[4:]
                ],
            }
            for i in range(10)
        ],
    }
