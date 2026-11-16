"""Start the frozen native service with the private six-event template."""

import runpy

from goodvision_harness import event_sensitivity, videojev_contract
from questions import DEFAULTS, scene_question_template

event_sensitivity.EVENT_SENSITIVITY_DEFAULTS.clear()
event_sensitivity.EVENT_SENSITIVITY_DEFAULTS.update(DEFAULTS)
videojev_contract.scene_question_template = scene_question_template
runpy.run_module("goodvision_harness.videojev_service", run_name="__main__")
