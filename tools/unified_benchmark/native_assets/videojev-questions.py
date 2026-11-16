"""Private six-event VideoJev choice template for the 120-question run."""

from goodvision_harness.event_sensitivity import sensitivity_policy_text
from goodvision_harness.videojev_contract import _PERSON_GUIDANCE, _WINDOW_GUIDANCE
from goodvision_harness.videojev_core import OPTIONS

LABELS = ("fight", "smoking", "fall", "weapon", "bag_knife", "bag_alcohol_bottle")
DEFAULTS = {label: "standard" for label in LABELS}


def scene_question_template(*, labels, sensitivities=None):
    if tuple(labels) != LABELS:
        raise ValueError("the private run requires the fixed six labels in order")
    if sensitivities and any(sensitivities.get(label, "standard") != "standard" for label in LABELS):
        raise ValueError("the private run requires standard sensitivity")
    policy = sensitivity_policy_text(
        {label: "standard" for label in LABELS[:4]},
        smoking_standard_rule=(
            "Smoking STANDARD: require a clearly visible cigarette, cigar, pipe "
            "or vaping device and a compatible smoking action; a hand near the "
            "mouth alone is insufficient."
        ),
        include_output_guidance=False,
    )
    policy += (
        "\n\nBag knife: positive only when a recognizable knife is visibly exposed "
        "in a bag or being taken out of a bag. Do not infer hidden contents."
        "\n\nBag alcohol bottle: positive only when a bottle recognizable as an "
        "alcohol bottle is visibly exposed in a bag or being taken out of a bag. "
        "Do not infer hidden contents or the contents of an unidentifiable bottle."
    )
    before = policy + '\nEvidence metadata: {"source_times": '
    after = "}" + _WINDOW_GUIDANCE + _PERSON_GUIDANCE
    return tuple({
        "id": "scene/" + label, "task": "choice", "options": list(OPTIONS),
        "query_parts": (before, after + f"\nFor the whole visible scene, is {label} present under that policy?"),
    } for label in LABELS)
