# Current test202 inputs and interpretation

Default protocol: `unified-frame-202-v1`. It expands test membership. Keep
validation24, exact 1 FPS sampling, current-image API prompts, native windows,
three total attempts, parsing, threshold selection and aggregation unchanged.
The older `unified-frame-120-v2` remains an explicit preflight option; old bundles
must keep their original source snapshot and must not be edited or relabelled.

Current202 also bounds the 0,1,2,... sampling grid by the last decoded video frame
timestamp relative to container start. Some new MP4 durations extend a few
milliseconds past an integer second without a frame at/after that second.
Preflight records the excluded tail times and retained times per video in
`media-audit.json`; all retained frames must still decode successfully. This
boundary repair changes the rules/source fingerprints, not the frozen input files.
Historical bundles keep their original code and fingerprint.

## Acquire and verify

Keep the [five original private inputs](README.md#get-the-frozen-inputs), including
the original120 manifest/samples and split map. They establish validation24 and
prove that the original120 subset has not changed. Add these **three** files from
the [single current dataset folder](https://drive.google.com/drive/folders/1kOyH59vxBp3uppyEoeyMjTMUa_CalTdh)
to the same private `--inputs` directory (eight JSON files total):

| File | Source | SHA-256 |
|---|---|---|
| `manifest_202.json` | [Download](https://drive.google.com/file/d/1Hj8TjxeIeN9AH5bIZjww4BbjieojMz6S/view) | `d5c019130d672afd40638a6ca51c1c84ca8e8c7fad0dbe3faf493187002bb11d` |
| `samples.json` | [Download](https://drive.google.com/file/d/1xzpCEgKtoQ26NbNOMc-85KxPiogjIzXr/view) | `bfb685074787eeb449e2b8c6d89ad6cb055a861363cdb4afae898d5c93631545` |
| `videos.json` | [Download](https://drive.google.com/file/d/1cJhdVSiiVBOId6M6cybBJctNyJbS4e6u/view) | `71a9e6b4b1e59ba59cc584c874b99ae2907ab10dc4459fb251fbc54d0f31cf2c` |

Preserve raw bytes and filenames; do not overwrite `test-samples.json` with
`samples.json`. Download videos using the folder's `download_videos.py` and
`videos.json` following its README. Put the **192 test videos** in `--media/test/`
and retain the **24 validation videos** in `--media/validation/`. Video hashes
come from the frozen manifests; the URL registry is not proof of file integrity.
The old separate82 folder is no longer the acquisition entrypoint. A newer Drive
revision that fails these pins requires an explicit dataset revision, not a hash bypass.

## Membership and reference limits

| Cohort | Questions | Distinct videos | Role / reference status |
|---|---:|---:|---|
| Validation | 24 | 24 | Unchanged; four base events, 3 yes + 3 no per event; reviewed |
| Original test | 1–120 | 110 | Original questions, answers and video hashes unchanged; reviewed |
| Robot single views | 121–202 | 82 | All test; inherited labels, **pending individual single-view review** |

Combined event counts are **fight32 / smoking22 / fall62 / weapon66 /
bag_knife10 / bag_alcohol_bottle10**. The additional82 are fall36 + weapon46.
They come from **41 paired clips, only 3 original recording groups**; they are
not82 independent recording sessions. Both views and all related cuts stay in
test. Do not add the original41 stereo clips again or move views to validation.

Preflight retains `draft_pending_user_review` and `pending_view_review`; it does
not mark the new labels human-reviewed. It checks the frozen source groups and
media hashes against validation, and writes a derived `split-audit.json` without
altering `validation_split.json`. Unknown training/development/actor/scene overlap
and prior test use remain limitations. Passing these checks does not prove label
correctness, a fresh held-out test, or new GPU/API runtime readiness.

## Reporting and calibration

The following counts describe the baseline; the optional supplement below expands
only test membership and its reported denominators.

Headline is **correct / 202**, including questions with missing evidence in the
denominator. A question contributes correct only when all planned responses are
valid. Complete evidence aggregates any yes → yes, otherwise any no → no, otherwise
unknown. Valid unknown is not a transport failure; final unknown earns no correct
answer against binary reference labels. Missing units/questions remain separate.

Reports include **original120** and **robot82** cohort counts/accuracies and six
event breakdowns. Both cohort reports retain their full planned denominators.
The robot82 and combined202 results carry the inherited-label limitation.
Use the original120 subset when comparing with old120 runs:202 changes the
class mix as well as the footage, so its headline delta is not a model-only change. The
API `base100` field continues to mean only original base questions1–100; it must
not silently become182. Bag20 is separate from Weapon.

Tune Luna/CLIP/SigLIP2 only on complete validation24 and freeze their artifact
before test. Do not tune to the new82 test labels. Bag remains uncalibrated at
Luna0.5 / margin0; boolean/argmax models have no numerical threshold.
Report protocol, dataset hashes and the review limitation with any future sheet
update. A `Dataset all120` header cannot receive a202 headline: use the same-run
original120 subset there, or agree on changing the target block to202 before
publishing. Do not create columns or change shared tables merely by running preflight.


## Optional test supplement

`preflight --test-supplement /private/path/test-supplement.json` appends questions
after original202. The current bag supplement adds questions203–222 on ten videos,
for **222 questions / 202 test videos**, with event counts **32/22/62/66/20/20**.
Its labels are assistant annotations, not human-confirmed. Sources and acquisition
links are in the [benchmark skill](../../.claude/skills/event-benchmark/SKILL.md).
The combined questions/video lists aid acquisition; they do not replace any of the
eight frozen inputs. No private labels or media belong in this repository.

The separate JSON document requires `schema: "event-test-supplement-v1"`,
`split: "test"`, a nonempty `dataset` identity and a nonempty `samples` list.
Each row has a consecutive integer `number` starting at203, a basename `filename`,
lowercase media `sha256`, an existing `event`, binary `proposed_answer`, nonblank
`human_review` provenance text and `source_video_group` (or `source_group`). Both
group keys, when supplied, must agree. Review text records the supplied provenance;
validation does not establish human approval or label correctness.

One video may have different event questions, but repeated video/event pairs,
conflicting media identities and identical bytes under different names are rejected.
Names and hashes cannot overlap original test or validation; source groups cannot
overlap validation, including the optional bag validation manifest. Related cuts
may share a group within test. Validation membership and threshold selection remain
unchanged. Never tune on these appended test questions.

Preflight retains the raw supplement and its SHA-256, question membership and review
provenance. Bundle verification ties the test count, cohort and appended references
to that retained input. Reports use correct / **all planned questions**, including
missing evidence, with separate **original120**, **robot82** and **test_supplement**
cohorts (120/82/20 for this supplement). Original API `base100` and `bag20` keep their
100/20 denominators. For other supplements, use the recorded counts; never hardcode
222. Historical baseline bundles keep their original source snapshots and counts.
