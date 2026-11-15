# Current test202 inputs and interpretation

Default protocol: `unified-frame-202-v1`. It expands test membership. Preserve
the 24 original validation questions, exact 1 FPS sampling, current-image API prompts, native windows,
three total attempts, parsing, threshold selection and aggregation unchanged.
The older `unified-frame-120-v2` remains an explicit preflight option; old bundles
must keep their original source snapshot and must not be edited or relabelled.

The current [ten-round batch](BATCH.md) additionally requires the existing bag
validation supplement for every model: 44 validation questions on 34 videos.
The eight snapshots below do not contain that separate private manifest.

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
the current private dataset package provided by the benchmark maintainer
to the same private `--inputs` directory (eight JSON files total):

| File | SHA-256 |
|---|---|
| `manifest_202.json` | `d5c019130d672afd40638a6ca51c1c84ca8e8c7fad0dbe3faf493187002bb11d` |
| `samples.json` | `bfb685074787eeb449e2b8c6d89ad6cb055a861363cdb4afae898d5c93631545` |
| `videos.json` | `71a9e6b4b1e59ba59cc584c874b99ae2907ab10dc4459fb251fbc54d0f31cf2c` |

Preserve raw bytes and filenames; do not overwrite `test-samples.json` with
`samples.json`. Download videos using the folder's `download_videos.py` and
`videos.json` following its README. Put the **192 test videos** in `--media/test/`
and retain the **24 original validation videos** in `--media/validation/`.
For the current batch, add the ten original bag validation clips there and pass
the frozen supplement and its independent digest to preflight. Video hashes
come from the frozen manifests; the URL registry is not proof of file integrity.
The old separate82 folder is no longer the acquisition entrypoint. A newer Drive
revision that fails these pins requires an explicit dataset revision, not a hash bypass.

## Membership and reference limits

| Cohort | Questions | Distinct videos | Role / reference status |
|---|---:|---:|---|
| Original validation | 24 | 24 | Unchanged; four base events, 3 yes + 3 no per event; reviewed |
| Bag validation supplement | 20 | 10 | Required for every model in the current batch; retain manifest reference provenance |
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

For the current batch, tune all six Luna/CLIP/SigLIP2 thresholds only on complete
validation44 and freeze their artifact before test. Do not tune to the new82
test labels. Only explicit historical validation24 runs leave bag uncalibrated
at Luna0.5 / margin0; boolean/argmax models have no numerical threshold.
Report protocol, dataset hashes and the review limitation with any future sheet
update. A `Dataset all120` header cannot receive a202 headline: use the same-run
original120 subset there, or agree on changing the target block to202 before
publishing. Do not create columns or change shared tables merely by running preflight.
