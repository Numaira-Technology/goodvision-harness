# Legacy v2.1 benchmark (explicit selection only)

This checkout includes the v2.1 controlled runner and the explicit
`--legacy-baseline` route. Use the matching frozen inputs and source fingerprints.
Historical artifacts still require their retained source snapshot; never remove
required flags or edit hashes to make a different snapshot run.

The default current202 route is **unified-frame-202-v1** in
[the current skill](../SKILL.md) and
[the unified guide](../../../../tools/unified_benchmark/README.md).
The two protocols have different sampling, parsers and source identities; do not
use the commands below as a shortcut for the current202 run.

For this explicitly selected route, use `tools/event_benchmark/SPEC.md`
(v2.1) and `CONTROLLED_RUN.md`. Historical v1 outputs are not v2 evidence.
Commands below run from the repository root, within the user's authorized scope.

## Selected v2.1 workflow (matching inputs required)

1. **Freeze the inputs before scoring.** Identify the candidate, dataset, reviewed
   manifest, media directories and output path. Verify manifest review status,
   samples hash, question/label identities and media hashes. The runner does this
   before requests. Do not change a draft status to bypass review.
   Read each manifest's actual status and fingerprint; `manifest_120.json`
   includes 100 base-event questions plus 20 separate bag-search questions.
   Question count is not unique-video count. Larger sets require their own reviewed
   manifest. Never assume a colleague's Desktop path exists.
   `--preflight` checks inputs offline without credentials or inference. Keep its
   pending-provider/hardware/authorization items distinct from input validity.
2. **Run accuracy from the repository root**, only after authorization for calls:
   ```sh
   python tools/event_benchmark/run_eval.py --candidate luna \
     --samples /path/to/samples.json --manifest /path/to/reviewed-manifest.json \
     --media /path/to/media1 /path/to/media2 \
     --provider-config /path/to/frozen-provider.json \
     --split-manifest /path/to/validation_split.json --split test \
     --interval 1.0 --workers 12 --out outputs/event-benchmark/luna-v2.json
   ```
   Repeat with qwen and gemini on identical inputs. Capture the source commit and
   result-file SHA-256 alongside the runner's samples/manifest/media/runner/spec
   hashes. Keep generated data in ignored `outputs/`, not tracked `results/`.
3. **Measure latency separately**, with the same input and question per candidate:
   ```sh
   python .claude/skills/event-benchmark/scripts/latency_bench.py \
     --candidate luna --video /path/to/representative-clip.mp4 \
     --provider-config /path/to/the-same-frozen-provider.json \
     --concurrency 12 --rounds 6 --out outputs/event-benchmark/lat-luna.json
   ```
   This is single-frame, single-predicate request latency, not product latency or
   twelve-camera throughput. Inspect failures and completion status before reporting.
4. **Report** using [the report fields](report-format.md), including coverage and provenance.
   Use `bench-sheet-contribute` only when publishing is requested. Its default
   preparation is an offline preview, with no sheet write.

## Frozen baseline rules

- Same questions from `samples.json`, same media, 1 fps centered frame schedule,
  720px width, one independent request per frame; output envelope fixed by backend.
- Language: complete JSON boolean. Luna: finite numeric probability in [0,1].
  No truthiness coercion of strings/numbers, no invalid response treated as negative.
- Clip score = maximum of **all scheduled valid frames**; `score >= 0.5` is yes.
  Any missing frame or invalid answer after three total attempts makes the clip an
  error, even if another frame was positive.
- Accuracy is correct/all questions, per event and overall. Errors and missing
  answers stay in the denominator and are reported separately, never as negatives.
- `--dense` or changed cadence is a separately labeled sensitivity experiment.
  Tuning is a separate variant; see `bench-threshold-tune`.
- Dataset growth changes the dataset fingerprint and comparison cohort. It does not
  itself change the sampling spec; rerun every compared candidate on the same set.
- Provisional labels may be described as historical/exploratory evidence only, never
  as reviewed v2 results. No new official run until the manifest is reviewed.
- No negatives means specificity unavailable, not zero false positives.

## Other model classes

`run_eval.py` currently supports language and Luna decision backends. Native
VideoJev and CLIP have different protocols. Read [model classes](model-classes.md)
before importing their results; shared formatting does not establish a controlled
comparison. Do not coerce margins or native verdicts into Luna probabilities.
