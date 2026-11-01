# Additional safety events

The existing feature selection, per-event sensitivity, JSON/SSE output, temporal
confirmation and evidence archive also support:

| Feature | Meaning | Evidence |
| --- | --- | --- |
| `ppe_violation` | A person lacks a reflective safety vest or a safety helmet, or both | A visible uncovered head or visible torso without reflective safety clothing; occlusion alone is insufficient |
| `ground_litter` | Discarded rubbish is visible on the ground/floor | Waste and ground contact must be visible; held objects, bin contents and stored tools/materials are excluded |

Select either feature in the console, or include it in the existing `features`
request array. Both default to standard sensitivity. High sensitivity retains
the same minimum visible-evidence requirements for these two events.

Per-feature endpoints follow the existing `/v1/results/{feature}/latest` and
`/v1/results/{feature}/events` routes. Existing routes and JSON fields are retained.

PPE violations follow the affected person. Ground litter is a scene condition:
its temporal `subject_ids` is empty (the archive uses the existing `scene` target
type with identifier `scene-ground-litter`) and it can be confirmed with nobody present.
It clears on repeated negative observations, never because people leave.
The scene-state result uses an object labelled `scene` with a full-frame extent
and `event_scope: scene`; this is not a localized litter bounding box. Direct
grounded image results retain actual litter boxes. Existing confirmation and
clearing thresholds, retention and video evidence handling remain in use.
