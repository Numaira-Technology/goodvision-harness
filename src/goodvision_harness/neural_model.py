"""Native image/text recognition client for the neural workflow block."""

from .neural_calibration import CORE_EVENTS, calibrated_choice, sampling_contract
from .videojev_model import VideoJevInspector
from .vlm_independent_model import IndependentVerdict


class NeuralInspector(VideoJevInspector):
    model_name = "openai/clip-vit-large-patch14"
    display_name = "Neural network recognition"
    metric_prefix = "neural_"

    def _payload(self, job):
        # Stateless requests also survive service restarts and multiple clients.
        self._template_registered = False
        self._neural_frame_count = len(job.frames)
        return super()._payload(job, sampling=sampling_contract(self.config))

    def _decode(self, response, expected, labels):
        if response.get("model") != self.model_name:
            raise ValueError("recognition service model mismatch")
        rows = response.get("raw")
        if not isinstance(rows, list):
            raise ValueError("invalid recognition choices")
        fields = {
            "score_kind",
            "score",
            "threshold",
            "image_index",
            "frame_scores",
            "thresholds_sha256",
            "sampling",
        }
        calibrated = [
            row for row in rows if isinstance(row, dict) and fields & row.keys()
        ]
        if not calibrated:
            return super()._decode(response, expected, labels)
        if (
            len(rows) != len(expected)
            or any(not isinstance(row, dict) for row in rows)
            or [row.get("id") for row in rows] != [row["id"] for row in expected]
        ):
            raise ValueError("recognition choices do not match requested events")
        calibrated_ids = {"scene/" + label for label in CORE_EVENTS}
        legacy = [row for row in expected if row["id"] not in calibrated_ids]
        verdicts = (
            {
                verdict.event_label: verdict
                for verdict in super()._decode(
                    {
                        **response,
                        "raw": [row for row in rows if row["id"] not in calibrated_ids],
                    },
                    legacy,
                    tuple(row["id"].removeprefix("scene/") for row in legacy),
                )
            }
            if legacy
            else {}
        )
        hashes = set()
        for question, row in zip(expected, rows, strict=True):
            label = question["id"].removeprefix("scene/")
            if label not in CORE_EVENTS:
                if fields & row.keys():
                    raise ValueError("unexpected calibrated event")
                continue
            index = calibrated_choice(
                row,
                label,
                getattr(self, "_neural_frame_count", None),
                sampling=sampling_contract(self.config),
            )
            hashes.add(row["thresholds_sha256"])
            if (
                row.get("task") != "choice"
                or type(row.get("answer_index")) is not int
                or row["answer_index"] != index
                or row.get("answer") != question["options"][index]
            ):
                raise ValueError("calibrated choice disagrees with its threshold")
            positive = None if index == 2 else index == 0
            verdicts[label] = IndependentVerdict(
                label,
                (),
                positive,
                None,
                (),
                ("scene_only",),
            )
        if len(hashes) != 1:
            raise ValueError("inconsistent threshold artifacts")
        self.last_response_diagnostics = {
            "choices_count": len(expected),
            "calibrated_scores": calibrated,
        }
        return tuple(verdicts[label] for label in labels)
