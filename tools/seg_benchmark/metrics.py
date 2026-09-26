"""Instance matching and explicit quality/latency accounting."""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_instances(truth, predicted, threshold):
    ious = np.zeros((len(truth), len(predicted)), dtype=float)
    for i, gt in enumerate(truth):
        for j, mask in enumerate(predicted):
            union = np.count_nonzero(gt | mask)
            ious[i, j] = np.count_nonzero(gt & mask) / union if union else 0
    # Prioritize maximum count above threshold; IoU only breaks count ties.
    score = (ious >= threshold) * (min(ious.shape, default=0) + 1) + ious
    a, b = linear_sum_assignment(score, maximize=True)
    return [
        (int(i), int(j), float(ious[i, j]))
        for i, j in zip(a, b, strict=True)
        if ious[i, j] >= threshold
    ]


def boundary_f1(a, b, tolerance=3):
    kernel = np.ones((3, 3), np.uint8)

    def edge(mask):
        mask = mask.astype(np.uint8)
        return mask - cv2.erode(
            mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )

    ea, eb = edge(a), edge(b)
    if not ea.any() or not eb.any():
        return 0.0
    da = cv2.distanceTransform(1 - ea, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    db = cv2.distanceTransform(1 - eb, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    recall = float(np.mean(db[ea > 0] <= tolerance))
    precision = float(np.mean(da[eb > 0] <= tolerance))
    return 2 * recall * precision / (recall + precision) if recall + precision else 0.0


def polygons_to_masks(people, width, height):
    masks = []
    for person in people:
        mask = np.zeros((height, width), np.uint8)
        if not person:
            raise ValueError("Each person needs at least one polygon")
        for polygon in person:
            points = np.asarray(polygon, dtype=float)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or len(points) < 3
                or not np.isfinite(points).all()
                or (points < 0).any()
                or (points[:, 0] >= width).any()
                or (points[:, 1] >= height).any()
            ):
                raise ValueError("Invalid annotation polygon coordinates")
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        if not mask.any():
            raise ValueError("Empty annotation mask")
        masks.append(mask.astype(bool))
    return masks


def score_frame(truth, predicted, iou_threshold):
    pairs = match_instances(truth, predicted, iou_threshold)
    return {
        "gt_people": len(truth),
        "matched": len(pairs),
        "false_positives": len(predicted) - len(pairs),
        "matched_iou_sum": sum(p[2] for p in pairs),
        "boundary_f1_sum": sum(
            boundary_f1(truth[i], predicted[j]) for i, j, _ in pairs
        ),
    }


def replay_delay(timestamp, cost_ms, previous_finish=None):
    """Single-lane FIFO simulation, not a live Harness/display measurement."""
    started = max(
        timestamp, previous_finish if previous_finish is not None else timestamp
    )
    finished = started + cost_ms / 1000
    return finished, (finished - timestamp) * 1000


def summarize(rows, budget_ms, target, expected_frames):
    times = [r["pipeline_ms"] for r in rows]
    quality = [r for r in rows if r.get("quality") is not None]
    gt = sum(r["quality"]["gt_people"] for r in quality)
    matched = sum(r["quality"]["matched"] for r in quality)
    on_time = sum(
        r["quality"]["matched"] for r in quality if r["simulated_ready_ms"] <= budget_ms
    )
    negative = [r for r in quality if r["quality"]["gt_people"] == 0]
    complete = len(rows) == expected_frames and len(quality) == expected_frames
    coverage = on_time / gt if gt else None
    curve = {}
    for budget in sorted({50, 100, 200, 500, 1000, 1500, 4000, budget_ms}):
        hits = sum(
            r["quality"]["matched"]
            for r in quality
            if r["simulated_ready_ms"] <= budget
        )
        curve[str(budget)] = hits / gt if gt else None
    return {
        "frames": len(rows),
        "expected_frames": expected_frames,
        "annotated_frame_observations": len(quality),
        "all_frames_reviewed": complete,
        "gt_person_observations": gt,
        "p50_pipeline_ms": float(np.percentile(times, 50)) if times else None,
        "p95_pipeline_ms": float(np.percentile(times, 95)) if times else None,
        "p99_pipeline_ms": float(np.percentile(times, 99)) if times else None,
        "p95_simulated_ready_ms": float(
            np.percentile([r["simulated_ready_ms"] for r in rows], 95)
        )
        if rows
        else None,
        "mask_recall": matched / gt if gt else None,
        "effective_coverage": coverage,
        "mean_iou_all_gt": sum(r["quality"]["matched_iou_sum"] for r in quality) / gt
        if gt
        else None,
        "mean_boundary_f1_all_gt": sum(r["quality"]["boundary_f1_sum"] for r in quality)
        / gt
        if gt
        else None,
        "false_positive_instances": sum(
            r["quality"]["false_positives"] for r in quality
        ),
        "negative_frame_false_positive_rate": sum(
            r["predicted_people"] > 0 for r in negative
        )
        / len(negative)
        if negative
        else None,
        "coverage_by_simulated_budget_ms": curve,
        "passes_target": coverage >= target if complete and gt else None,
    }
