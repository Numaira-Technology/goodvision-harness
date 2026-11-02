# Provisional base-4 diagnostics — base4-provisional-v1

`run_eval.py --provisional` is an explicitly unreviewed diagnostic mode. It accepts
draft binary labels for fight, smoking, fall and weapon, optionally filtered with
`--events`. It does not verify a reviewed manifest. An explicit `--manifest` is an
error in this mode; use the normal frozen mode to verify one.

The runner checks unique question numbers, nonempty binary question text, yes/no
labels, and the presence of unambiguous media files before any provider request.
It records hashes of the samples and media it reads; these fingerprints record
the inputs and do not establish human review or agreement with a frozen source.

Frame sampling, backend response parsing, three total attempts with one-second
failure delays, 90-second request timeout, complete-frame requirement, clip-max
aggregation and the 0.5 gate follow the v2 runner mechanics. Default cadence is
1 fps; a changed interval or dense sampling must be reported separately. Scores
are lossless. Errors are excluded from accuracy and counted alongside attempted
and scored questions; the console marks the entire report PROVISIONAL.

Output uses `spec_version=base4-provisional-v1`, `frozen=false`, `provisional=true`,
`manifest_sha256=null`, the selected `scored_events`, and this document's hash.
These outputs cannot enter v2 threshold selection/evaluation or a reviewed v2
comparison table. Completing review later does not retroactively qualify a run:
freeze reviewed manifests and collect fresh compatible evidence.

Bag-search questions have a separate yes/no/unknown answer space. Neither this
mode nor frozen v2 implements their answer, aggregation or metric contract.
`--events bag_knife`/`bag_alcohol_bottle`, non-binary labels, and questions requesting
an unknown answer are rejected. Do not change unknown to no or remove the third
answer from the question to make it fit the binary runner. A combined 120-question
metric requires a separately reviewed protocol; this mode covers only its base-4
questions. Existing bag archives are historical binary experiments, not evidence
that the three-valued task or v2 acceptance requirements were satisfied.
