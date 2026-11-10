# Native CLIP threshold artifacts

The neural service can apply four externally selected cosine-margin thresholds
to `fight`, `smoking`, `fall` and `weapon`. Supply a UTF-8 JSON file with
`goodvision-neural --cache-dir /path/to/cache --thresholds /path/to/artifact.json`
or set `NEURAL_THRESHOLDS_FILE` in the service process environment. An explicit
CLI path takes precedence. Missing or invalid files stop startup before loading
model weights. Without a configured file, the original prompts and pair choices
remain unchanged. PPE and ground-litter decoding retain their existing contracts.

Upgrade the service and harness client together before enabling thresholds. Older
clients validate answers against the probability argmax and can reject a valid
threshold-based negative or unknown answer. The updated client still accepts
unconfigured services' legacy responses.

The file must have exactly these fields:

| Field | Required value |
| --- | --- |
| `schema_version` | Integer `1` |
| `model` | `openai/clip-vit-large-patch14` |
| `revision` | `32bd64288804d66eefd0ccbe215aa642df71cc41` |
| `prompts` | Object mapping the four events to the exact two strings below |
| `preprocessing` | `rgb-bicubic-proportional-pad-224-fill-123-117-104-clip-processor-v1` |
| `aggregation` | `max_all_images` |
| `comparison` | `>` |
| `thresholds` | Exactly the four event keys; each value is a finite number in `[-2, 2]` or `null` |
| `provenance` | Exactly the four fields listed below |

The `prompts` object uses ordered positive/negative arrays:

| Event | Positive | Negative |
| --- | --- | --- |
| `fight` | Two people are physically fighting, punching or kicking each other. | People are walking, standing or talking peacefully. |
| `smoking` | A person is smoking a cigarette. | A person is eating food or drinking from a cup. |
| `fall` | A person is losing balance and falling to the ground. | A person is standing, walking or sitting steadily. |
| `weapon` | A person is holding a gun or a knife. | A person is holding an ordinary object such as a phone. |

`provenance` must contain `validation_manifest_sha256`,
`validation_scores_sha256` and `selection_protocol_sha256` as lowercase 64-digit
hex hashes, plus `source_commit` as a lowercase 40-digit Git commit. These identify
the reviewed split/labels/media/source groups, complete lossless validation scores,
and frozen selection protocol used to produce the thresholds. Retain those
artifacts separately. The service validates the format and inference compatibility;
it cannot verify evidence it has not received, the reviewer identity, source
isolation, or detection accuracy. Hashes are identifiers, not signatures or proof
of review. Do not invent values to make an unreviewed artifact load.

Duplicate keys, extra/missing fields, boolean/string thresholds, nonfinite values,
incompatible metadata and files larger than 64 KiB are rejected. The accepted file
bytes are SHA-256 hashed once and its values remain fixed for the process lifetime.
Changes require restarting and passing warmup. Bare event-to-number maps are
intentionally rejected because they cannot identify their inference contract or
validation evidence.

For each requested core event, the response includes `score_kind: cosine_margin`,
the maximum `score`, all `frame_scores` in request order, the first winning
`image_index` (zero-based), `threshold`, and `thresholds_sha256`. Every supplied
image must have finite valid output, even when another image already supports a
positive choice. A score strictly greater than the threshold is positive; equality
is negative. A `null` threshold produces unknown. Missing or invalid outputs fail
the request and are never replaced with negative or unknown detections.

`probabilities` retains the actual positive/negative softmax pair for that winning
image with a third zero entry. It remains a diagnostic pair when the chosen answer
is negative or unknown; its argmax does not override the threshold. It is not an
accuracy estimate or event confidence. The client validates the complete window,
winner, artifact identity and threshold choice, and preserves PPE derived unknowns.

Validation must match the full service preprocessing: incoming RGB JPEGs,
proportional bicubic padding to 224 by 224 with fill `(123, 117, 104)`, then the
processor from the pinned model revision. Upstream image selection, JPEG resizing,
window cadence and source grouping also need to match the reviewed validation
protocol; this file does not reconfigure or certify them. Center-cropped validation
is incompatible with the full-frame production path. Use the pinned neural extra
versions; offline tests exercise PyTorch 2.8.0 and Transformers 5.14.1 interfaces
with tiny random CPU weights, which does not establish GPU or model accuracy.

This change supplies no production thresholds, reviewed validation split, calibration
results or held-out accuracy claim. The separate benchmark/threshold-selection
review requirements still apply; no evidence hold is waived by enabling this loader.
