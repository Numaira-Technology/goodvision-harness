# Temporal VLM request timing

The temporal adapter emits two correlated records on the existing server log
channel. Both carry the bounded `stream_session_id` and `request_id`; the service
pool emits a third dispatch record with the same identifiers. Durations
are monotonic seconds. Missing or null measurements mean unavailable, not zero.
The adapter keeps its existing non-streaming protocol and total HTTP deadline.

## Connection ownership and prewarming

The service keeps up to four temporal client owner threads alive across runs.
Each thread creates, uses and closes its own inspector, asyncio loop and HTTP
connection pool. A run borrows independent leases; stopping that run retires its
leases without closing reusable connections. Pending obsolete work is cancelled,
and results remain attached to their original run. Idle owners release completed
jobs and their images rather than retaining them until the next request.

When a run starts, each owner prewarms a new client in parallel with the remaining
startup work. Prewarming sends a credential-free, body-free HEAD to the configured
endpoint with a total budget of at most two seconds, bounded by the inference
timeout. It strips endpoint credentials, query parameters and fragments and does
not follow redirects. A 401 or 405 can establish a reusable transport connection;
it does not establish model health. Failed prewarming does not block later
inference or count as a failed model request. If inspection arrives before
prewarming finishes, it waits on that owner and its pool dispatch timing includes
the wait. Synchronous client initialization itself is not interruptible.

An unchanged configuration reuses clients across runs. Preparing a run after at
least 45 seconds of owner inactivity requests another prewarm, before the HTTP
pool's 60-second idle expiry. There is no periodic heartbeat while the service is
idle. The remote server can close a connection earlier, so reuse is best effort.
Model, endpoint, credentials, inference options, pool size and proxy/TLS
environment changes replace the clients; local execution setting updates also
invalidate them. An active request finishes on its old owner before that client
closes. Each owner admits at most one active and one pending request. Application
shutdown closes the pool with a bounded join; an in-flight HTTP operation can
finish and close its transport afterward. Process restarts cannot preserve TCP
connections.

| Record | Fields and meaning |
|---|---|
| `vlm_connection_warmup_timing` | `total_seconds` and `outcome` (`transport_ready`, `timeout`, `failed`, `closed`); excludes model inference, carries no run identifier |
| `vlm_pool_dispatch_timing` | `pool_queue_wait_seconds`: submission to actual inspector invocation, including owner availability and unfinished prewarming; excludes the scene scheduler's queue |

Pool dispatch time is outside `vlm_inspection_timing.total_seconds`. The first
request's initialization, TCP and TLS cost may have moved into startup prewarming;
zero initialization time on that request alone does not mean the cost disappeared.
Compare startup-to-result latency as well as warm request latency.

## Whole inspection: `vlm_inspection_timing`

| Field | Meaning |
|---|---|
| `inspector_lock_wait_seconds` | Wait for this inspector's serialization lock; not the scene scheduler queue |
| `request_prepare_seconds` | Validate frames, resize/re-encode images, construct messages and authorization headers |
| `client_initialize_seconds` | Initialize the event loop and HTTP pool when necessary; does not include TCP/TLS connection |
| `verdict_parse_seconds` | Parse and validate event judgments after the HTTP response completes |
| `total_seconds` | Whole inspection, including lock wait, local work, and HTTP |
| `requested_event_count`, `returned_event_count`, `verdict_count` | Requested labels and aggregate result coverage; successful HTTP is not proof of complete coverage |
| `empty_verdicts` | Successful response yielded no event judgments; do not equate this with explicit false judgments |
| `outcome`, `error_code` | Inspection outcome, including validation failures after a successful HTTP exchange |

## HTTP exchange: `vlm_http_timing`

Existing TCP/TLS, request-header/body writes, response-header wait, response-body
read, byte/token counts and outcome fields remain available. Added measurements:

| Field | Meaning |
|---|---|
| `request_build_seconds` | HTTPX request construction and serialization |
| `dispatch_to_first_transport_event_seconds` | Dispatch to the first supported transport trace event; includes scheduling/pool/startup overhead, not a pure pool-queue measurement |
| `request_body_sent_at_seconds` | Offset from HTTP start when body transmission completes |
| `response_first_non_whitespace_seconds` | Offset from body-read start of the first non-JSON-whitespace byte; not first model token |
| `http_first_non_whitespace_seconds` | Same observation measured from HTTP start |
| `upload_to_first_non_whitespace_seconds` | From completed upload to that observation, when upload completion was traced |
| `response_leading_whitespace_bytes` | Leading SP/TAB/CR/LF bytes, often gateway keepalive data |
| `response_max_chunk_gap_seconds` | Largest observed interval to a nonempty body chunk, including initial wait |
| `response_chunk_samples` | First eight nonempty chunks: body-read offset, decoded byte count, and whitespace-only flag |
| `response_chunk_samples_dropped` | Number of later chunks excluded from the bounded sample list; total counters still include them |
| `response_json_decode_seconds` | Decode the response envelope, including failed decoding |
| `response_close_seconds` | HTTP response cleanup |
| `cached_prompt_tokens` | Provider-reported cached-token count, only when a valid nonnegative integer is supplied |
| `generation_ref` | SHA-256 of a bounded provider generation identifier, for joining diagnostic records without exposing the raw identifier |

Chunk timings and partial byte counts survive a response-body timeout. An error
HTTP status is recorded before the body is rejected. Oversized body chunks are
rejected before they are copied into the body buffer or diagnostic samples.
Log samples never contain response text, image bytes, headers or credentials.

The response-header phase can overlap upstream inference; a gateway can emit
headers and whitespace before the final JSON. Neither first response chunk nor
first non-whitespace byte is a reliable model TTFT for this non-streaming API.
`response_read_seconds` includes waiting for bytes, not just wire transfer.
Trace events do not separately expose DNS, remote queueing or GPU prefill.

## Optional upstream metadata lookup

The diagnostic audit can retain a generation identifier in memory and query the
configured API's generation metadata **after** all timed inference calls. It
joins numeric provider metadata to `generation_ref`. Such lookups are not added
to the production request path and do not send video frames again.

The [generation metadata API](https://openrouter.ai/docs/api/api-reference/generations/get-generation)
provides fields including `latency`, `generation_time`, token counts and provider
attempts when available. Treat returned fields as provider-defined measurements,
not disjoint phases to add together. The provider's
[performance-metric documentation](https://openrouter.ai/docs/guides/community/for-providers#11-performance-metrics)
states generation time includes upstream fetch latency, TTFT and streaming time,
so it is not pure GPU execution time. A 404 metadata lookup is recorded as
unavailable; it does not change a successful inference into an error.

Compare both actual serving provider and output coverage when comparing runs.
The same public route can select different upstream providers. An empty array or
one-label answer cannot establish equivalent performance to four explicit
judgments merely because it finishes sooner.
