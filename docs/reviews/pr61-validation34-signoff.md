# PR 61 validation sign-off packet

Status: **proposed, pending Ashley's sign-off. KEEP UNMERGED.**
This file is not a runner input and not a frozen threshold artifact.
This agent did not set `human_reviewed` or `user_reviewed` on any new record.

## What was measured

- All 34 validation videos were downloaded from the Drive ids in `validation34.json` (folder `14QePPZqvuOkF2gCOHSFICdFx--QO0rkX`) and SHA-256 hashed locally. All 34 match that index.
- Planned 1 FPS observations on those 34 files: **327** (ffprobe). Total duration about 312 seconds.
- Pinned containers remeasured and matched: `validation-samples.json`, `validation_split.json` (`a0375a6a846fe964846a58d94ed68a5a2e18c61a238b0f36940d4c2a83f51b2c`), `manifest_202.json`, `samples.json`, `videos.json`.
- `manifest_validation_24.json` and `bag-validation.json` container digests were **not** remeasured from saved bytes. Their per-file media hashes match the downloaded files. The hash-verified sidecar records the published validation-24 container pin `41a3ef7cadc16e5a4e511ce779688e07811e450aba168bb62b33d0f42bfa7365`.
- Test identity check used the pinned 202 manifest and samples. The 192 test video bytes were **not** re-downloaded or re-hashed.
- `tune_threshold.validate_manifest` on this proposed file raises `a human_reviewed manifest is required`.

## Counts

- Validation candidate: 34 videos / 44 questions = original 24 clips (questions 1–24) plus 10 bag clips (questions 25–44).
- The 10 bag clips are validation-only in the Drive index. They do not appear in the 202-question test manifest.
- Formal test set remains 192 videos / 202 questions.
- Recorded review labels, copied and not changed: questions 1–24 `user_reviewed` (already on the frozen 24 manifest); questions 25–44 `assistant_visual_review_no_individual_human_labels_supplied`.

## Proposed source groups

| Group | Files |
|---|---|
| `Ashley-bag-20261111-black` | IMG_2895.MOV, IMG_2896.MOV, IMG_2897.MOV, IMG_2898.MOV |
| `Ashley-bag-20261111-green` | IMG_2892.MOV, IMG_2893.MOV, IMG_2894.MOV |
| `Ashley-bag-20261111-yellow` | IMG_2899.MOV, IMG_2900.MOV, IMG_2901.MOV |
| `HMDB:nice_smoking_girl` | nice_smoking_girl_smoke_h_nm_np1_le_med_0.mp4 |
| `HMDB:smoking_1` | smoking_1_smoke_h_cm_np1_ri_goo_0.mp4 |
| `HMDB:smoking_video` | smoking_video_smoke_h_nm_np1_fr_goo_1.mp4 |
| `SCVD:train:w001` | w001_converted.avi |
| `SCVD:train:w003` | w003_converted.avi |
| `SCVD:train:w004` | w004_converted.avi |
| `UBI:f_182_0_0_1_0` | ubi_F_182_0_0_1_0.mp4 |
| `UBI:f_210_1_0_0_0` | ubi_F_210_1_0_0_0.mp4 |
| `UBI:f_69_0_0_0_0` | ubi_F_69_0_0_0_0.mp4 |
| `UBI:n_481_0_0_1_0` | ubi_N_481_0_0_1_0.mp4 |
| `UBI:n_537_0_1_1_0` | ubi_N_537_0_1_1_0.mp4 |
| `UBI:n_724_1_0_1_0` | ubi_N_724_1_0_1_0.mp4 |
| `URFD:adl-24` | adl-24-cam0.mp4 |
| `URFD:adl-25` | adl-25-cam0.mp4 |
| `URFD:adl-26` | adl-26-cam0.mp4 |
| `URFD:adl-28` | adl-28-cam0.mp4 |
| `URFD:adl-29` | adl-29-cam0.mp4 |
| `URFD:adl-30` | adl-30-cam0.mp4 |
| `URFD:adl-32` | adl-32-cam0.mp4 |
| `URFD:adl-33` | adl-33-cam0.mp4 |
| `URFD:adl-34` | adl-34-cam0.mp4 |
| `URFD:fall-02` | fall-02-cam0.mp4 |
| `URFD:fall-28` | fall-28-cam0.mp4 |
| `URFD:fall-29` | fall-29-cam0.mp4 |

Bag groups come from `bag-validation.json`: three visible bag setups (green, black, yellow). That file says props and setting recur inside a setup, and that recording sessions, locations and actors were not verified. These groups are proposals.

The 24 original groups are the ones already stored on the hash-verified sidecar. That sidecar sets `independent_owner_group_review` to false.

## Isolation

- No validation filename or SHA-256 appears in pinned `manifest_202.json`.
- Test bag filenames are `IMG_1185`–`IMG_1189` and `IMG_2729`–`IMG_2733`. Validation bags are `IMG_2892`–`IMG_2901`. No shared filename.
- Exact `source_video_group` strings do not equal any test `source_group`.
- Training membership for the evaluated models is **not established**. Development history is **incomplete**; the sidecar says existing test material has known previous development/evaluation use.
- Actor and scene isolation is **not proven**. URFD rooms can overlap. `SCVD:train:*` is the source dataset's split label, not proof these clips were in a model training set.
- Recording-number coincidence, different files:
  - Validation question 20 `w003_converted.avi` `7dd2b7ba147eefa8cffb72d780546c83882a0a2cd2855c03fa92d40a63b344d8` group `SCVD:train:w003`, and test question 46 `t_w003_converted.avi` `eaa2fb25c437c3d5cb3538dafae0101f9f544b2ac35f7e729fc813dc329b47f8` group `SCVD_v3_t_w003`.
  - Validation question 21 `w004_converted.avi` `d80e2e73491b703fb38a68e6333949db5e551092514b422f865238805dfab188` group `SCVD:train:w004`, and test question 87 `t_w004_converted.avi` `bb3364ca79dcadb08573a184493e64e6efbe39cece0c00e8d1c8f2529154436d` group `SCVD_v3_t_w004`.
  Hashes differ. Whether they share a person or scene is not established.
- Pinned `manifest_202.json` is still `draft_pending_user_review`. Questions 121–202 remain `pending_view_review` in that frozen snapshot. The 2026-11-12 index says Pengyu confirmed those 82 labels; this agent did not edit the snapshot to match that note.

## Why calibration and the frozen table did not run

No model call, GPU rental or API request was made. Spend against the $2 cap: **$0.00**. No Vast instance was created, so there is no instance id and no invoice. `VAST_API_KEY`, `R2_*`, `OPENROUTER_API_KEY` and `GOOGLE_DRIVE_ACCESS_TOKEN` are unset on this machine. Drive reads used the connected Drive account. A direct unauthenticated download returned a Google sign-in page.

Public RTX 4090 list prices read on 2026-11-14 disagree (about $0.14 to $0.54 per GPU-hour depending on the aggregator). That is not this account's offer and not a bill. Validation is 327 one-FPS observations. Test frame count is unknown because the 192 test videos were not downloaded. A short 4090 rental might fit under $2 at those list prices. It was not started.

The current CLIP/SigLIP2 selector (`tools/clip_benchmark/scoring.py`) midpoint-selects only fight, smoking, fall and weapon. Bag thresholds stay 0. It does not emit six midpoint thresholds. That rule was not changed. Runner v2.1 (`tools/event_benchmark`) also requires a `human_reviewed` manifest and provider receipts before `select` can freeze an artifact.

## Files, hashes, proposed groups

| # | File | Bytes | SHA-256 | Proposed group |
|---:|---|---:|---|---|
| 1 | `ubi_F_182_0_0_1_0.mp4` | 334536 | `9eeb940f23ced044fea89af5885ed987d3c3d1f617dda304ec87100d373e037f` | `UBI:f_182_0_0_1_0` |
| 2 | `ubi_F_210_1_0_0_0.mp4` | 271426 | `cc12339290435a89eb4f341574f19a17fd220ce0203e7e1d0372e4dbebf322aa` | `UBI:f_210_1_0_0_0` |
| 3 | `ubi_F_69_0_0_0_0.mp4` | 822329 | `59435e1d8364cd0a6f482d6ae700fbf276bbbda181b91db719e54d88ee45842e` | `UBI:f_69_0_0_0_0` |
| 4 | `ubi_N_481_0_0_1_0.mp4` | 277344 | `b73ccd63bffe426fd99229deb27be51922ce0d18c618782671b11f1123d0b1c0` | `UBI:n_481_0_0_1_0` |
| 5 | `ubi_N_537_0_1_1_0.mp4` | 425714 | `e7a6904b95f481ff1ba696ba7de4ab24d634f3ac5a6bf3f69a7d24940ac0d34b` | `UBI:n_537_0_1_1_0` |
| 6 | `ubi_N_724_1_0_1_0.mp4` | 288637 | `05f04f52c7142c8a156e3a006e3107dc330a726b7c4a12956c123fcde5c20edb` | `UBI:n_724_1_0_1_0` |
| 7 | `smoking_video_smoke_h_nm_np1_fr_goo_1.mp4` | 595574 | `a696d0d26898dfdf016216f5a2ab2ff2de9cf55717610234c5de75776491d518` | `HMDB:smoking_video` |
| 8 | `nice_smoking_girl_smoke_h_nm_np1_le_med_0.mp4` | 171686 | `caf6d3927d09937b048ba7036a2e93378e9486b5fd21e2b114a2e31885c7169f` | `HMDB:nice_smoking_girl` |
| 9 | `smoking_1_smoke_h_cm_np1_ri_goo_0.mp4` | 466326 | `cc57b94527f82193f76f66614b41b403f047a964ba141dcd982d6c1a765ea126` | `HMDB:smoking_1` |
| 10 | `adl-24-cam0.mp4` | 33792 | `e9c5714da6a0e30766f5590d5dfe8f219d188ff40a64a20edbefe6b56ace39c5` | `URFD:adl-24` |
| 11 | `adl-25-cam0.mp4` | 38701 | `c2774517592fb0f12036efd93ac25aa8becb12e72f162646e439cee80beb5e0b` | `URFD:adl-25` |
| 12 | `adl-30-cam0.mp4` | 222499 | `a879572f12ae2b27b522cf3fa5d974e8402b6e7fc10fc962d0a85e4bbea168e5` | `URFD:adl-30` |
| 13 | `fall-02-cam0.mp4` | 93187 | `0ce7885bb313d11476b493f5f6d3baf56553ab7fb5d7b299b3ddbc52b9bcc595` | `URFD:fall-02` |
| 14 | `fall-28-cam0.mp4` | 51200 | `a1ee3f0b992a687678774c975ef6673a0688b88e4ed55ce33356c949c30a1917` | `URFD:fall-28` |
| 15 | `fall-29-cam0.mp4` | 66050 | `ac00ba6d13e123eae5a3ebbcf7f8faed04901245891b382fd51cca09b5fc6eb0` | `URFD:fall-29` |
| 16 | `adl-26-cam0.mp4` | 68033 | `79dacd23811230a18870d5155f70cdd289fdf9736ba897797d35a83165d468d8` | `URFD:adl-26` |
| 17 | `adl-28-cam0.mp4` | 36355 | `287ef0020d55bf86654976f7c747702462b95d6263d5429bdcdf2f4ec2511578` | `URFD:adl-28` |
| 18 | `adl-29-cam0.mp4` | 50417 | `805afe93e5c0393debdb575611b41e8c38801d2f0a17ccc411273d70188ba626` | `URFD:adl-29` |
| 19 | `w001_converted.avi` | 810978 | `f25c3f02c258ebe5ffc1082a61114f220f0b25171ba5194fea28b33d5294f719` | `SCVD:train:w001` |
| 20 | `w003_converted.avi` | 707964 | `7dd2b7ba147eefa8cffb72d780546c83882a0a2cd2855c03fa92d40a63b344d8` | `SCVD:train:w003` |
| 21 | `w004_converted.avi` | 1229990 | `d80e2e73491b703fb38a68e6333949db5e551092514b422f865238805dfab188` | `SCVD:train:w004` |
| 22 | `adl-32-cam0.mp4` | 119105 | `f8ec44e3a0e83087fd94543f15236df3bca430e2a7924cfffbedeedde90bc438` | `URFD:adl-32` |
| 23 | `adl-33-cam0.mp4` | 115535 | `570e5b2a5c4960ad9d356f92173641ee7b25ce4a56b1a2b65f92ff2b369a219c` | `URFD:adl-33` |
| 24 | `adl-34-cam0.mp4` | 134295 | `1be342c918bfdb43e0811a5d07fbd6430d385b376b5b2f2f0acb50a3909dc14d` | `URFD:adl-34` |
| 25 | `IMG_2892.MOV` | 3866463 | `b321cbe2700e4b98657c75c609c55458a562c62d3307dc9e273ed628de400f86` | `Ashley-bag-20261111-green` |
| 27 | `IMG_2893.MOV` | 3848510 | `ebb174a88ec4c65835a11da41c1c0dd8df502dfab6fabd49a1ecd352bee9e776` | `Ashley-bag-20261111-green` |
| 29 | `IMG_2894.MOV` | 3792758 | `d50b5d4f3a4286fbb61b358cedcc3c02d337f839df9afc982d20a0f558e2414b` | `Ashley-bag-20261111-green` |
| 31 | `IMG_2895.MOV` | 3343413 | `ad634bc73c367da3fbcb163aa2b1f4962e8ac07a8f6a2767dd143d889710bca1` | `Ashley-bag-20261111-black` |
| 33 | `IMG_2896.MOV` | 4331461 | `1b075f001c7a13fc92f034f9f46328a14f50cc8b93c2a933cc08e880ab64a893` | `Ashley-bag-20261111-black` |
| 35 | `IMG_2897.MOV` | 5148352 | `4a0931730542360c8e5bfceaf44dfee816e7877fb6e74b7d0616c9e4444b0234` | `Ashley-bag-20261111-black` |
| 37 | `IMG_2898.MOV` | 3436950 | `29e0e326da87a54e466902d3f1702493bb68ba43bfef6de77f7c3d6e42962e49` | `Ashley-bag-20261111-black` |
| 39 | `IMG_2899.MOV` | 5080503 | `73fbfaf7c5ac22777e2e0798fe1b229c77d51556dec504b76e9e3afa0408fdec` | `Ashley-bag-20261111-yellow` |
| 41 | `IMG_2900.MOV` | 5126102 | `ae5be57829cd69cf9b077f3b255348701c6b9b19f895365a4ff0cfa5c5da4acc` | `Ashley-bag-20261111-yellow` |
| 43 | `IMG_2901.MOV` | 4698394 | `e78371697d64ffd238e69d9b45e495be35b9ed15aaef617d7e0cfbcc3f171bac` | `Ashley-bag-20261111-yellow` |

Question text, answers and recorded review labels are in `validation34-proposed-manifest.json` next to this file.

## What Ashley still has to confirm

1. These 34 files and hashes are the validation media she means.
2. The proposed groups above, including the three bag-color groups, are the groups she wants. Same-group clips may share props and setting.
3. Questions 25–44 (bag knife and bag alcohol, 10 each, both yes and no present) may be marked `user_reviewed` only by her. They are assistant visual labels with no individual human label sheet. This packet does not mark them.
4. The 10 bag clips stay in validation and never enter the 202-question test score.
5. The w003/w004 recording-number coincidence with test questions 46 and 87 is acceptable, or those clips must be regrouped. Hashes already differ.
6. Training, development, actor and scene isolation stays unproven. Signing the file list does not clear the KEEP UNMERGED hold by itself.
7. After that sign-off, a real CLIP and SigLIP2 run still needs `VAST_API_KEY` and R2 credentials. The current selector will calibrate four base thresholds and leave both bag thresholds at 0 unless the threshold rule is changed in a separate decision.

