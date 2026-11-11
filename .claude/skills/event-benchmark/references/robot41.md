# Reviewed robot41 supplement (explicit selection only)

This checkout includes the reviewed robot41 manifest/samples and downloader.
Verify their fingerprints and the downloaded media before a separately authorized
run. The video folder alone does not establish the dataset identity.

Use this separate cohort only when requested. It does not replace or extend the
current202 exam, which includes82 single views instead of these41 stereo clips.
Do not count both sets. For the default workflow return to [the skill](../SKILL.md).
The instructions below apply only to this explicitly selected cohort.

Use `tools/event_benchmark/datasets/samples_robot41.json` with
`tools/event_benchmark/datasets/manifest_robot41.json`. These are the 41 clips
Pengyu reviewed on November 3, numbered 121–161; they do not replace or silently
extend the original 120-question cohort. Videos remain in the
[team Drive folder](https://drive.google.com/drive/folders/132wI2o9nLno7a_IGw7IG7zLJabBqTYxW).

| Category | Reference yes | Reference no | Questions |
|---|---:|---:|---:|
| fall | 9 | 9 | 18 |
| weapon | 11 | 12 | 23 |

Preserve the supplied questions and answers. Here `weapon` asks about visibly
holding or carrying a knife or weapon; it does not require an attack. The confirmed
beverage bottle and pen are `no`. Walking away, returning and crossing laterally
are separate routes; two near-camera knife displays remain separately identified.
Lateral weapon coverage still lacks two yes and one no clips for the requested
three-of-each quota. Clips share three source recordings: do not split cuts from
the same `source_group` between threshold tuning and evaluation. There are no
reviewed natural-onset times or person-ID labels; do not claim reaction-time or
track-ID accuracy from this supplement.

Prepare media on the machine that will run inference; on Vast, download there
directly rather than staging videos through a laptop. Use an existing Google
account with access to the folder. The downloader accepts a read-only Drive OAuth
token in `GOOGLE_DRIVE_ACCESS_TOKEN`, or `--links` pointing to a private JSON map
of filenames/Drive IDs to temporary HTTPS download URLs. A connected Drive fetch
can supply those URLs. Obtain them after environment preparation, refresh only
expired entries, and keep the map and credentials outside Git. Private Drive
view links are not anonymous media download URLs.

```sh
python tools/event_benchmark/download_media.py --output outputs/robot41/media \
  --links /path/to/private-temporary-links.json
python tools/event_benchmark/download_media.py --output outputs/robot41/media --check-only
```

The second command uses the same input loader as the evaluator and should report
41 questions, 41 videos, `fall: 18`, `weapon: 23`, and zero model requests. It does
not start a model. For an authorized accuracy run, use the explicit cohort:

```sh
python tools/event_benchmark/run_eval.py --legacy-baseline --candidate qwen \
  --samples tools/event_benchmark/datasets/samples_robot41.json \
  --manifest tools/event_benchmark/datasets/manifest_robot41.json \
  --media outputs/robot41/media --interval 1.0 --workers 12 \
  --out outputs/event-benchmark/robot41-qwen.json
```

`--legacy-baseline` explicitly uses the original default-provider v2.1 baseline
without the validation24/test120 sidecar. It preserves the centered sampling,
strict parser and fixed gate, and records `v2.1-legacy-unpinned-lossless-v1`.
Do not mix it with controlled provider/split options or use its output for
validation threshold selection. The [controlled legacy workflow](legacy-v2.1.md) still requires
the provider configuration and split sidecar. Add `--preflight` for offline checks.

Report this as a separate robot41 cohort with full denominators 18, 23 and 41.
The existing Sheet's 50/100/120 blocks are not robot41 result cells. Availability
and reviewed labels do not establish model recognition accuracy; no model result
is supplied with these annotations. Other backends retain their existing runner
boundaries in [model classes](model-classes.md).
