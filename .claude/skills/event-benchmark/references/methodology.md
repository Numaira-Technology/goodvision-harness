# Methodology (frozen exam — swap the candidate, not the exam)

This is a controlled exam: the only thing allowed to change is the model under test. Prompt, sampling,
threshold, and scoring are byte-identical for every candidate. Changing any of them changes the exam —
bump a version and re-run every candidate.

The authoritative source lives in the repo; don't copy the rules here and let them drift:
- Spec: `tools/event_benchmark/SPEC.md` (exam spec v1)
- Runner: `tools/event_benchmark/run_eval.py`
- History: `tools/event_benchmark/RESULTS.md` + `results/`

Below is a quick reference. SPEC.md wins on any detail.

## Frozen items

- **Prompt**: each sample's own `question` field from `samples.json`, with the trailing `Answer yes or no.`
  stripped at runtime. The question is the same for every model; only the output envelope is fixed per class.
- **Scored events**: currently fight / smoking / fall / weapon. Clip-level (did it happen in the clip), not
  track_id / person selection.
- **Three model classes**, output fixed within a class and not aligned across classes (full contract in
  `model-classes.md`):
  - **language** (qwen = baseline / gemini, chat completions): prompt + frame → JSON boolean
    `{"answer": true/false}`, 0/1 per frame.
  - **decision**: Luna decisions (`noul` question type, 0–1 probability, calibratable on a held-out set);
    VideoJev-4B (native multi-label present/absent/insufficient verdict, uncalibrated option scores, no threshold).
  - **threshold** (CLIP / SigLIP2, colleague's `tools/clip_benchmark`, not wired into run_eval): per-frame
    **margin = cos(positive) − cos(negative)**, `> its own threshold` → yes. A margin is not a probability,
    so never borrow 0.5; its untuned baseline is threshold = 0.
  - Real runner locations, per-class cadence (VLM 1fps · CLIP 1 frame/clip @0.8s · VideoJev 2 frames/window
    @0.8s), data sync, and honesty flags: see `model-classes.md`.
- **Sampling**: 1 fps (`--interval 1.0`, frozen cadence), 720px wide, one frame one independent call.
  - 0.25 fps (`--interval 4`) was only a production compromise when VLM throughput couldn't keep up; it is
    not a hard requirement, and eval + production are unified at 1 fps.
  - `--dense` is a capability-ceiling comparison (≤14 frames), for sampling-sensitivity checks only, not the
    headline number.
- **Decision**: 0–1 score per frame, clip = max over frames ("positive if any frame is positive"),
  `≥ gate 0.5` → yes. Gate 0.5 is uniform across every event and model, and is never tuned on the eval set.
- **Run params**: temperature 0, chat max_tokens 30, 90s timeout, 3 retries, `--workers 12` (mirrors the
  12-way product; affects speed only, not scoring).

## The dataset can grow; the spec does not

Sample count can grow (50 → 120 → ~250), and growing it does not change the exam. When expanding:
- Freeze the file list + a SHA-256 per clip (`--checksums`); no relabeling after you've seen scores.
- Add negatives to a positives-only event before you talk about its false-positive rate / specificity.
- Tune thresholds / prompts only on a held-out subset, never on the main eval clips.
- A new question type (e.g. repackaging bag_knife / bag_alcohol_bottle, whose answer space includes
  `unknown`) is **changing the exam**: version it separately and define the unknown-handling rule first;
  don't fold it into the binary main table.

## Where the dataset lives (moves with the project; confirm before a run)

Current merged set: `~/Desktop/benchmark_media_test/media_extra_50/samples.json` (120 questions), videos
spread across:
- `media50/media/media/` (original 50)
- `media_extra_50/新增基础视频_50段/`
- `media_extra_50/翻包视频_10段/` (repackaging, not base-4)

`run_eval.py --media` takes several directories and finds clips by filename across them.
