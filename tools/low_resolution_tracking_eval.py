"""Evaluate saved actor observations with PersonRegistry, entirely offline.

No inference, camera, network, enrollment or webhook calls are made. Optional
CAVIAR XML must already exist locally; its geometry is matched one-to-one to
actors independently on each frame before subject continuity is evaluated.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goodvision_harness.contracts import (  # noqa: E402
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.grounding import ModelActor, ObservationBundle  # noqa: E402
from goodvision_harness.person_registry import PersonRegistry  # noqa: E402

CAVIAR_XML_URL = (
    "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA2/"
    "EnterExitCrossingPaths2front/feecp2gt.xml"
)


def box_iou(left, right):
    intersection = max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1e-12, left_area + right_area - intersection)


def one_to_one_matches(actors, truth, threshold):
    """Maximize match count, then total IoU, without using either identity."""
    scores = [[box_iou(a["bbox"], g["bbox"]) for g in truth] for a in actors]

    @lru_cache(None)
    def search(index, used):
        if index == len(actors):
            return 0, 0.0, ()
        best = search(index + 1, used)
        for gt_index, score in enumerate(scores[index]):
            if used & (1 << gt_index) or score < threshold:
                continue
            count, total, pairs = search(index + 1, used | (1 << gt_index))
            candidate = count + 1, total + score, ((index, gt_index, score), *pairs)
            if candidate[:2] > best[:2]:
                best = candidate
        return best

    return search(0, 0)[2]


def read_caviar(path):
    root = ET.parse(path).getroot()
    if root.tag != "dataset" or root.get("name") != "EnterExitCrossingPaths2front":
        raise ValueError("XML is not the selected CAVIAR crossing sequence")
    frames = {}
    for frame in root.findall("frame"):
        objects = []
        for obj in frame.findall("./objectlist/object"):
            box = obj.find("box")
            if box is None or obj.findtext("appearance") == "disappear":
                continue
            xc, yc, width, height = (float(box.get(k)) for k in ("xc", "yc", "w", "h"))
            if width <= 0 or height <= 0:
                continue
            objects.append(
                {
                    "gt_id": obj.get("id"),
                    "bbox": [
                        (xc - width / 2) / 384,
                        (yc - height / 2) / 288,
                        (xc + width / 2) / 384,
                        (yc + height / 2) / 288,
                    ],
                    "appearance": obj.findtext("appearance"),
                }
            )
        frames[int(frame.get("number"))] = objects
    if set(frames) != set(range(485)):
        raise ValueError("Unexpected CAVIAR timeline: expected frames 0..484")
    return frames


def replay_actors(rows):
    """Pass real saved actors through the production prepare_analysis method.

    Neutral carrier objects expose each actor's returned subject ID. They have
    no positive event judgment and do not create events or modify the actors.
    """
    session = f"offline-{rows[0]['clip_id']}-{rows[0]['long_side']}"
    registry = PersonRegistry(session, "offline-evaluation")
    subject_names = {}
    observations = []
    for sequence, row in enumerate(rows, start=1):
        timestamp = float(row.get("actual_timestamp", row["timestamp"]))
        frame_index = int(row.get("actual_source_frame_index", sequence))
        actors = tuple(
            ModelActor(
                a["actor_id"],
                NormalizedBBox(*(a["bbox"][key] for key in ("x1", "y1", "x2", "y2"))),
                a.get("confidence", 1.0),
            )
            for a in row.get("actors", ())
        )
        width, height = row["original_dimensions"]
        carrier = AnalysisResult(
            message_id=row["key"],
            source_id="offline-evaluation",
            stream_session_id=session,
            sequence=sequence,
            emitted_at="2026-09-10T00:00:00Z",
            frame=FrameReference(
                frame_index, timestamp, "2026-09-10T00:00:00Z", width, height
            ),
            judgments=(EventJudgment("diagnostic", False, 1.0, "Actor replay"),),
            objects=tuple(
                GroundedObject(
                    actor.actor_id,
                    "person",
                    actor.bbox,
                    actor.confidence,
                )
                for actor in actors
            ),
            model_name="saved-output",
            model_latency_seconds=0,
        )
        prepared = registry.prepare_analysis(
            carrier,
            ObservationBundle(
                actors=actors,
                object_actor_ids=tuple((a.actor_id,) for a in actors),
            ),
        )
        observed = []
        for actor, obj in zip(actors, prepared.objects, strict=True):
            subject = obj.attributes.get("subject_id")
            if subject is not None:
                subject = subject_names.setdefault(
                    subject, f"s{len(subject_names) + 1}"
                )
            observed.append(
                {
                    "actor_id": actor.actor_id,
                    "subject_id": subject,
                    "bbox": [
                        actor.bbox.x1,
                        actor.bbox.y1,
                        actor.bbox.x2,
                        actor.bbox.y2,
                    ],
                    "confidence": actor.confidence,
                }
            )
        observations.append(
            {
                "key": row["key"],
                "timestamp": timestamp,
                "source_frame_index": frame_index,
                "original_dimensions": row["original_dimensions"],
                "input_error": row.get("error"),
                "actors": observed,
            }
        )
    return observations


def score_continuity(frames, *, max_gap=3.0, gt_ids=None):
    counts = defaultdict(int)
    last_by_gt, last_by_subject = {}, {}
    details = []
    previous_truth, previous_time = set(), None
    for frame in frames:
        timestamp = frame["timestamp"]
        visible = {g["gt_id"] for g in frame["truth"]}
        if gt_ids is not None:
            visible &= set(gt_ids)
        current = {m["gt_id"]: m for m in frame["matches"] if m["gt_id"] in visible}
        counts["gt_observations"] += len(visible)
        counts["matched_gt_observations"] += len(current)
        counts["missed_gt_observations"] += len(visible - current.keys())
        counts["associated_gt_observations"] += sum(
            match["subject_id"] is not None for match in current.values()
        )
        counts["unassociated_gt_observations"] += sum(
            match["subject_id"] is None for match in current.values()
        )
        counts["input_error_frames"] += bool(frame.get("input_error"))
        counts["frames"] += 1
        if previous_time is not None and timestamp - previous_time <= max_gap:
            counts["adjacent_gt_links"] += len(visible & previous_truth)
        for gt_id, match in current.items():
            subject = match["subject_id"]
            if subject is None:
                continue
            prior = last_by_gt.get(gt_id)
            if prior:
                if timestamp - prior["timestamp"] > max_gap:
                    counts["reappearances_after_expiry"] += 1
                else:
                    counts["evaluated_continuity_links"] += 1
                    if prior["subject_id"] != subject:
                        counts["subject_switches"] += 1
                        details.append(
                            {
                                "kind": "subject_switch",
                                "gt_id": gt_id,
                                "timestamp": timestamp,
                                "previous": prior,
                                "subject_id": subject,
                            }
                        )
                    else:
                        counts["preserved_links"] += 1
            prior_subject = last_by_subject.get(subject)
            if (
                subject is not None
                and prior_subject
                and timestamp - prior_subject["timestamp"] <= max_gap
                and prior_subject["gt_id"] != gt_id
            ):
                counts["wrong_inheritances"] += 1
                details.append(
                    {
                        "kind": "wrong_inheritance",
                        "subject_id": subject,
                        "timestamp": timestamp,
                        "gt_id": gt_id,
                        "previous": prior_subject,
                    }
                )
            last_by_gt[gt_id] = {"subject_id": subject, "timestamp": timestamp}
            if subject is not None:
                last_by_subject[subject] = {"gt_id": gt_id, "timestamp": timestamp}
        previous_truth, previous_time = visible, timestamp
    for key in (
        "gt_observations",
        "matched_gt_observations",
        "missed_gt_observations",
        "subject_switches",
        "wrong_inheritances",
        "evaluated_continuity_links",
        "preserved_links",
        "reappearances_after_expiry",
        "adjacent_gt_links",
    ):
        counts[key] += 0
    counts["gt_match_rate"] = (
        counts["matched_gt_observations"] / counts["gt_observations"]
        if counts["gt_observations"]
        else None
    )
    counts["continuity_rate_on_matched_links"] = (
        counts["preserved_links"] / counts["evaluated_continuity_links"]
        if counts["evaluated_continuity_links"]
        else None
    )
    return {"counts": dict(counts), "issues": details}


def gt_score(observations, truth, threshold):
    frames = []
    for observation in observations:
        index = observation["source_frame_index"]
        if (
            observation["original_dimensions"] != [384, 288]
            or abs(index / 25 - observation["timestamp"]) > 0.041
            or index not in truth
        ):
            return {"available": False, "reason": "source/XML alignment failed"}
        targets = truth[index]
        actors = observation["actors"]
        pairs = one_to_one_matches(actors, targets, threshold)
        matches = [
            {
                "actor_id": actors[a]["actor_id"],
                "subject_id": actors[a]["subject_id"],
                "gt_id": targets[g]["gt_id"],
                "iou": score,
            }
            for a, g, score in pairs
        ]
        frames.append(
            {
                **observation,
                "truth": targets,
                "matches": matches,
                "unmatched_actors": len(actors) - len(matches),
            }
        )
    return {
        "available": True,
        "iou_threshold": threshold,
        "all_annotated_people": score_continuity(frames),
        "foreground_people_0_1": score_continuity(frames, gt_ids={"0", "1"}),
        "unmatched_actor_observations": sum(f["unmatched_actors"] for f in frames),
        "frames": frames,
    }


def diagnostics(observations):
    seen, last_time = set(), None
    count = defaultdict(int)
    for frame in observations:
        timestamp = frame["timestamp"]
        count["frames"] += 1
        count["actors"] += len(frame["actors"])
        count["input_error_frames"] += bool(frame.get("input_error"))
        count["zero_actor_frames"] += not frame["actors"]
        count["source_gaps_over_3s"] += (
            last_time is not None and timestamp - last_time > 3
        )
        subjects = {a["subject_id"] for a in frame["actors"]} - {None}
        count["previously_seen_subject_observations"] += len(subjects & seen)
        seen.update(subjects)
        last_time = timestamp
    return {**count, "unique_subjects": len(seen), "ground_truth_available": False}


def geometry_controls(truth):
    """A separate perfect-box control isolates association from model error."""
    result = {}
    for step in (5, 10, 20, 40):
        records = []
        for index in range(200, 401, step):
            records.append(
                {
                    "key": f"gt-control-{index}",
                    "clip_id": "crossing",
                    "timestamp": index / 25,
                    "actual_timestamp": index / 25,
                    "actual_source_frame_index": index,
                    "long_side": 0,
                    "original_dimensions": [384, 288],
                    "actors": [
                        {
                            "actor_id": f"a{number}",
                            "bbox": dict(
                                zip(("x1", "y1", "x2", "y2"), obj["bbox"], strict=True)
                            ),
                            "confidence": 1.0,
                        }
                        for number, obj in enumerate(truth[index])
                    ],
                }
            )
        result[str(step / 25)] = gt_score(replay_actors(records), truth, 0.5)
    return {
        "scope": "GT boxes only; not real model or GPU tracking accuracy",
        "by_source_interval_seconds": result,
    }


def markdown(report):
    aligned = any(
        score["available"]
        for group in report["groups"]
        for score in group.get("ground_truth", ())
    )
    lines = [
        "# Offline actor continuity evaluation",
        "",
        f"Saved records: {report['records']} / {report['expected_records']}; "
        f"complete: {report['complete']}. No new model calls.",
        "",
        (
            "Ground truth is available for the crossing clip."
            if aligned
            else "No aligned ground truth is available; outputs are diagnostics only."
        ),
        "",
        "CAVIAR crossing uses official object IDs with frame-by-frame one-to-one "
        "IoU assignment, maximizing match count then overlap. Matching does not "
        "use persistent subject IDs. Foreground IDs 0/1 and all IDs 0/1/2 are "
        "reported separately. Persons smaller than a few pixels remain in the "
        "denominator. Subject changes after more than 3 s without matched "
        "observations are excluded from switch counts.",
        "",
        "| Long side | IoU | GT matched | GT missed | Unassociated matches | "
        "Matched links | "
        "Preserved | ID switches | Wrong inheritance |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group in report["groups"]:
        if group["clip_id"] != "crossing":
            continue
        for score in group.get("ground_truth", []):
            if not score["available"]:
                continue
            c = score["all_annotated_people"]["counts"]
            lines.append(
                f"| {group['long_side'] or 'original'} | {score['iou_threshold']} | "
                f"{c['matched_gt_observations']}/{c['gt_observations']} | "
                f"{c['missed_gt_observations']} | "
                f"{c['unassociated_gt_observations']} | "
                f"{c['evaluated_continuity_links']} | "
                f"{c['preserved_links']} | {c['subject_switches']} | "
                f"{c['wrong_inheritances']} |"
            )
    lines += [
        "",
        "This is sparse actor-assisted association at the saved sample times, "
        "not dense GPU tracking or face-identification accuracy. Missing actors "
        "reduce match coverage; continuity conditional on matched pairs must "
        "not be read as overall success. Actor/GT geometry matching includes "
        "low-confidence actors; matches without a registry subject ID are "
        "reported separately and cannot establish continuity. "
        "A subject ID switching may be a "
        "deliberate conservative reset. Wrong inheritance means that one "
        "subject ID maps to different annotated people within 3 s.",
        "",
        "Other clips have no aligned GT here: their actor/subject counts are "
        "diagnostics only, not accuracy. Recorded gaps over 3 s are expected "
        "expiry cases. This run sorts already completed responses by source "
        "time; it does not measure concurrency, inference latency or token cost.",
        "",
        f"GT source: {CAVIAR_XML_URL}",
    ]
    if report.get("geometry_control"):
        lines += [
            "",
            "## Separate GT geometry control",
            "",
            "This feeds perfect annotated boxes to the registry to isolate "
            "association failures. These are not model or GPU performance scores.",
            "",
            "| Interval (s) | Matched links | Preserved | Switches | "
            "Wrong inheritance |",
            "|---|---|---|---|---|",
        ]
        for interval, score in report["geometry_control"][
            "by_source_interval_seconds"
        ].items():
            c = score["all_annotated_people"]["counts"]
            lines.append(
                f"| {interval} | {c['evaluated_continuity_links']} | "
                f"{c['preserved_links']} | {c['subject_switches']} | "
                f"{c['wrong_inheritances']} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--caviar-xml", type=Path)
    parser.add_argument("--expected-records", type=int, default=222)
    parser.add_argument(
        "--gt-control",
        action="store_true",
        help="also run clearly separate perfect-box association controls",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for line in args.journal.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["key"]] = row
    truth = read_caviar(args.caviar_xml) if args.caviar_xml else None
    groups = defaultdict(list)
    for row in rows.values():
        groups[row["clip_id"], row["long_side"], row["quality"]].append(row)
    results = []
    for (clip_id, size, quality), records in sorted(groups.items()):
        records.sort(
            key=lambda r: (r.get("actual_timestamp", r["timestamp"]), r["key"])
        )
        observations = replay_actors(records)
        group = {
            "clip_id": clip_id,
            "long_side": size,
            "quality": quality,
            "diagnostics": diagnostics(observations),
            "observations": observations,
        }
        if clip_id == "crossing" and truth is not None:
            group["ground_truth"] = [
                gt_score(observations, truth, t) for t in (0.3, 0.5)
            ]
            group["diagnostics"]["ground_truth_available"] = any(
                score["available"] for score in group["ground_truth"]
            )
        results.append(group)
    report = {
        "records": len(rows),
        "expected_records": args.expected_records,
        "complete": len(rows) == args.expected_records,
        "mode": "saved actors, production PersonRegistry, source-time order",
        "new_model_calls": 0,
        "gt_source": CAVIAR_XML_URL if truth else None,
        "gt_geometry": "384x288 center-xywh converted to normalized xyxy; 25fps",
        "groups": results,
    }
    if args.gt_control and truth is not None:
        report["geometry_control"] = geometry_controls(truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
