"""Official Decision 3 choices on the shared prepared-window transport."""

import math

from .decision3_runtime import MODEL, REVISION
from .videojev_model import VideoJevInspector
from .vlm_independent_model import IndependentVerdict


class Decision3Inspector(VideoJevInspector):
    """Keep per-question unknown/errors separate while retaining valid siblings."""

    model_name = MODEL
    display_name = "Decision 3"
    metric_prefix = "decision3_"

    @staticmethod
    def _choice(answer, options):
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("invalid choice answer")
        if answer.get("error") is not None:
            raise ValueError("official question error")
        probabilities, choice = answer.get("probabilities"), answer.get("choice")
        if not isinstance(probabilities, dict) or set(probabilities) != set(options):
            raise ValueError("missing or unknown choice probabilities")
        values = [probabilities[option] for option in options]
        if (
            any(type(value) not in (int, float) or not math.isfinite(value)
                or not 0 <= value <= 1 for value in values)
            or not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-4)
            or choice not in options
            or choice != options[max(range(len(options)), key=values.__getitem__)]
        ):
            raise ValueError("invalid choice probability or selected option")
        confidence = answer.get("confidence")
        if confidence is not None and (
            type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("invalid choice confidence")
        return (True, False, None)[options.index(choice)], confidence

    def _decode(self, response, expected, labels):
        if response.get("model") != MODEL or response.get("revision") != REVISION:
            raise ValueError("unexpected Decision 3 model")
        self.last_raw_response = response
        self.last_question_errors = {}
        raw = response.get("raw")
        answers = raw.get("answers") if isinstance(raw, dict) else None
        answers = answers if isinstance(answers, dict) else {}
        by_id = {row["id"]: row for row in expected}
        verdicts = []
        for label in labels:
            question_id = "scene/" + label
            answer = answers.get(question_id)
            try:
                detected, confidence = self._choice(answer, by_id[question_id]["options"])
                issues = ("scene_only",)
            except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
                self.last_question_errors[question_id] = {"reason": str(error), "raw": answer}
                detected, confidence = None, None
                issues = ("scene_only", "decision3_question_error")
            verdicts.append(IndependentVerdict(label, (), detected, confidence, (), issues))
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if isinstance(usage, dict):
            for source, target in (("input_tokens", "prompt_tokens"),
                                   ("output_tokens", "completion_tokens")):
                value = usage.get(source)
                if type(value) is int and value >= 0:
                    self.last_http_timing[target] = value
        self.last_response_diagnostics = {
            "choices_count": sum(row["id"] in answers for row in expected),
            "model_match": "matched",
        }
        return tuple(verdicts)
