# Harness capability gap — why the eval set is 54, not 80

The 80-question diversity sample (`bench_v1_sample80/`) is the full, source-
balanced set. This harness can only answer **54** of them. The other **26 are
dropped** into the harness-answerable set `bench_v1_sample54/` because they fall
outside what the harness perceives — not because of a runner bug. This document
is the record for the eval side.

## Root cause

The harness is a **safety-event perception system**. Its scene/VLM engine detects
a fixed taxonomy of four events (see `config.yaml` → `rich_labeling.multi_event_prompt`):

> **fight · smoking · fall · weapon**

Person/vehicle tracking (SAM) additionally supports counting, location, and
count-over-time questions. Nothing in the harness detects general **behavioral /
activity events** (talking, reading, opening a door, entering a vehicle, turning,
shop entry, walking-vs-immobile, …).

## What the 26 dropped questions ask

They are meva/caviar semantic-event questions with no counterpart in the four
safety events:

| capability | dataset | n | why unanswerable |
| --- | --- | --- | --- |
| event_recognition | meva | 9 | events like `person_texts_on_phone`, `person_opens_facility_door`, `vehicle_turns_left/right`, `person_talks_to_person` — no detector |
| event_recognition | caviar | 6 | `walking`, `immobile`, `shop_enter`, `shop_reenter` — no detector |
| temporal_questions | meva | 7 | "event X begins at t, how long until it ends" — needs to detect **and time** an undetectable event |
| temporal_questions | caviar | 4 | same event-duration grounding on undetectable events |

Distinct event labels dropped: `meva:event_selection`, `caviar:event_selection`,
`meva:any_official_activity`, `meva:vehicle_turns_{left,right}`,
`meva:person_{talks_to_person,reads_document,texts_on_phone,opens_facility_door,
enters_scene_through_structure}`, `caviar:{walking,immobile,shop_enter,shop_reenter}`.

Feeding these anyway would make the runner answer `no` to every event question
(wrong on positives) and error on every temporal one — i.e. fabricated or empty
output. We drop them instead.

## The 54 the harness does answer

Full diversity is retained — all 9 datasets, all 4 capabilities, all 4 answer
formats — with meva/caviar contributing through **location_tracking** (person
tracking works regardless of event type) and event questions limited to the
**fight** family:

- location_tracking 17 · counting 16 · event_recognition 12 (fight) · temporal 9
  (3 fight-end + 6 MOT count-over-time)
- datasets: vscrowd 11, mot17 9, mot20 8, meva 7, ntu 6, caviar 4, rwf2000 3,
  ubi_fights 3, surveillance_camera_fight 3.

## What it would take to answer the 26 (not done)

- **event_recognition (15):** extend the VLM event prompt to the queried event
  per question — i.e. a generic per-question VLM call. This changes the
  evaluation contract (the runner deliberately reads structured detector output,
  not free-form VLM answers) and would need validation of that path.
- **event-duration temporal (11):** additionally requires VLM temporal grounding
  (start/end localization) of an event the model was not trained to detect —
  substantially harder and out of scope for the current harness.

Recommendation: report the 54 as the harness's coverage and score the 26 as
out-of-capability, rather than adjust the measurement to hide the gap.
