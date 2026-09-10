# Compact negative responses: 8080 demo validation (2026-09-10)

The current 8080 image includes compact negative responses, automatic selection by lowest median latency, and persistent manual route selection.

Three before/after requests used the same deterministic 320×180 synthetic route-probe image, corresponding frame indices, and saved route (public E; internal G). Each group reused one HTTP client; the first request included a cold connection. All six returned valid all-negative judgments. This small sequential comparison does not establish a live-camera latency SLA or detector accuracy; upstream Nitro routing and load can vary.

| Trial | Before seconds | After seconds | Before output tokens | After output tokens |
|---|---:|---:|---:|---:|
| 1 | 3.145 | 1.227 | 410 | 10 |
| 2 | 2.479 | 0.733 | 434 | 10 |
| 3 | 2.292 | 0.655 | 398 | 10 |

Median: 2.479s → 0.733s (70.4% lower in this sample). Output length fell from 398–434 tokens to 10 tokens. Input tokens increased from 1,742 to 1,901 because of the explicit compact-output instructions.

The compact wire response is `{"no_events":0.95}` with model-supplied confidence. The service expands it into complete per-event negative judgments and generic bilingual summaries, without another model request. Positive results keep the original evidence, bounding-box and output-budget requirements. Auxiliary actors are omitted on all-negative frames; local face recognition is independent.

Validation: 1,886 tests passed and 6 skipped using the existing development environment with FFmpeg on PATH. Focused lint and diff whitespace checks passed. The Docker build validated parsing and speed-first selection. Browser verification exercised a manual selection of public route A and confirmed the saved-selection message; regression tests cover persistence after runtime restart, active-session/probe guards, invalid route IDs and failed saves.

The subsequent live console route check completed all eight routes, each with
three valid responses. It selected and persisted public route A at 0.454s median,
the minimum across A–H (0.454, 1.041, 0.547, 1.219, 0.663, 0.679, 1.030, 0.662s).
The API and browser both confirmed the winner. The demo remained idle, matching
the session state immediately before deployment; safety input remains 320px and
face input 1280px on the longest side.
