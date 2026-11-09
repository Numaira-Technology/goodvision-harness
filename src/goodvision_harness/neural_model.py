"""Native image/text recognition client for the neural workflow block."""

import math

from .videojev_model import VideoJevInspector
from .vlm_independent_model import IndependentVerdict


class NeuralInspector(VideoJevInspector):
    model_name = "openai/clip-vit-large-patch14"
    display_name = "Neural network recognition"
    metric_prefix = "neural_"

    def _payload(self, job):
        # Stateless requests also survive service restarts and multiple clients.
        self._template_registered = False
        return super()._payload(job)

    def _decode(self, response, expected, labels):
        if response.get("model") != self.model_name:
            raise ValueError("recognition service model mismatch")
        rows = response.get("raw")
        if not isinstance(rows, list):
            raise ValueError("invalid recognition choices")
        calibrated = {row["id"]: row for row in rows if isinstance(row, dict)
                      and row.get("score_kind") == "cosine_margin" and "id" in row}
        if not calibrated:
            return super()._decode(response, expected, labels)
        if (len(rows) != len(expected) or any(not isinstance(row, dict) for row in rows)
                or [row.get("id") for row in rows] != [row["id"] for row in expected]):
            raise ValueError("recognition choices do not match requested events")
        legacy = [row for row in expected if row["id"] not in calibrated]
        verdicts = {v.event_label: v for v in super()._decode(
            {**response, "raw": [row for row in rows if row["id"] not in calibrated]},
            legacy, tuple(row["id"].removeprefix("scene/") for row in legacy),
        )} if legacy else {}
        for question in expected:
            row = calibrated.get(question["id"])
            if row is None:
                continue
            score, threshold = row.get("score"), row.get("threshold")
            if (type(score) not in (int, float) or not math.isfinite(score)
                    or threshold is not None and (type(threshold) not in (int, float)
                                                 or not math.isfinite(threshold))):
                raise ValueError("invalid calibrated cosine margin")
            positive = None if threshold is None else score > threshold
            index = 2 if positive is None else 0 if positive else 1
            probabilities = row.get("probabilities")
            if (not isinstance(probabilities, list) or len(probabilities) != 3
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or not 0 <= value <= 1 for value in probabilities)
                    or abs(sum(probabilities) - 1) > 1e-5):
                raise ValueError("invalid recognition probabilities")
            if (row.get("task") != "choice" or type(row.get("answer_index")) is not int
                    or row["answer_index"] != index or row.get("answer") != question["options"][index]):
                raise ValueError("calibrated choice disagrees with its threshold")
            label = question["id"].removeprefix("scene/")
            verdicts[label] = IndependentVerdict(label, (), positive, None, (), ("scene_only",))
        self.last_response_diagnostics = {"choices_count": len(expected),
                                          "calibrated_scores": list(calibrated.values())}
        return tuple(verdicts[label] for label in labels)
