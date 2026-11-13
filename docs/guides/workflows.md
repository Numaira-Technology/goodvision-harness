# Workflow canvas

Open **Workflows** in the console. The canvas fits the browser window; drag empty
space to pan, use the zoom controls or Ctrl/Command + wheel, and choose **Fit** to
see the graph. Existing live video, people library, event replay, output and
deployment pages remain available without reconnecting cameras.

## Build, test and apply

1. Create a workflow. Drag a **Video source** onto the canvas for each camera.
   The main container creates an isolated worker automatically when needed; no
   channel count or redeployment is required. Select each source to set its
   URL and mono/stereo layout, then **Save video source** and **Connect video**.
   The node and inspector display the shared live input; connection and
   disconnection are synchronized with the live workspace.
2. Add processing and output blocks by clicking the library or dragging them onto
   the canvas. Connect an output port to an input of the same type. Clicking an
   already-connected port pair again disconnects it; Undo restores the connection.
   Connect **Video source → Face recognition → Output**. Add a parallel
   **Workflow input → Every-minute summary → Output** path. Each summary instance
   has its own interval and instruction; the default is 60 seconds.
3. **Test** uses one current frame from a connected, stopped camera. It runs the
   configured face/event/summary adapters without publishing events, sending
   webhooks or changing the applied setup. Results identify their camera and
   source frame; synthetic results are marked. A single snapshot does not verify
   temporal confirmation, continuous tracking or event evidence.
4. Select an **Event recognition**, **Summary** or **Person tracking** block to
   configure its service directly in the node sidebar. Each service has its own
   save button; switching nodes preserves pending service edits. API credentials
   and route testing remain in **Deployment & settings**.
   Each camera follows only paths reachable from its own source; several sources
   may share processing blocks. Camera groups remain available for shared switches.
5. **Apply** submits the current graph and updates every camera group already
   using this workflow in one atomic operation. There is no separate Save stage.
   Bound video sources are assigned automatically when applying. A camera already
   owned by another workflow must be unassigned first. Removing a source disables
   its applied pipeline without disconnecting its video. For a legacy workflow
   with an unbound input, choose groups in the Apply dialog or leave the
   selection empty to apply the workflow configuration alone. Use **Camera group
   assignments…** to add groups later; existing assignments are always included.
   All affected cameras must be stopped and finished cleaning up. Invalid paths,
   unavailable cameras or a concurrent change reject the entire operation and
   keep the local edits. Apply never starts analysis or disconnects a preview.
6. Use **Start all analysis** in the live workspace. The coordinator installs the
   applied plans and checks connected cameras before starting. Cameras whose
   applied pipelines are all disabled are excluded. **Stop all analysis** stops
   processing while retaining video connections.

Edits stay local until Apply. Switching workflows preserves pending edits and
Undo history; no request is sent for individual keystrokes. Apply and Test require
complete source-to-output paths. A newly created workflow can be assembled locally
before it has a runnable graph.

Connecting video does not start analysis. Both views read the same camera
connection and decoded frames. If connection fails, both show the reason:
unreachable video server, signaling timeout, rejected request, invalid WHEP
response or no incoming media after negotiation. For a private-network URL,
the service host must be able to reach the video server; saving the URL alone
does not establish network access. WHEP also requires the negotiated media
address to be reachable after its HTTP signaling request succeeds.

If a deployment still has test mode enabled, the workflow displays a persistent
notice that analysis results are simulated. Stop analysis and select **Use real
analysis** to persist `dry_run: false` without saving unrelated service drafts or
starting a model. The next Start uses the configured real services; their usual
readiness and credential checks still apply.

## Camera and group switches

Camera groups display the applied graph read-only. Select **All cameras** for
group switches, or one camera for its `Inherit`, `On` and `Off` overrides. A group
or group block switched off is a hard gate: a camera's `On` cannot bypass it.
Changing the workflow requires every camera in every associated group to be
stopped, including cameras whose group or block switches are off. Starting,
running, testing, stopping, draining, cleanup or unavailable status locks edits
and Apply. The editor keeps pending changes and unlocks after the cameras stop;
viewing, navigating, copying, panning and zooming remain available. Group switches
also require their affected cameras to be stopped. Execution uses the shared
Start/Stop controls; Apply does not replace a running graph.

The live results pane displays camera-qualified outputs and recent summary
history. A summary uses the latest source snapshot at each interval, starting
after the first interval. It does not summarize a preceding video window. Each
summary has one request in flight, a timeout and bounded history. Errors are
reported on that block; other processing continues. Summary history is held in
memory for the current run, not an archive.

## Supported blocks and settings

| Block | Configuration and behavior |
| --- | --- |
| Video source | One camera per source, allocated on demand; URL, mono/stereo layout, Connect/Disconnect and shared input preview. Multiple sources are supported. |
| Person tracking → Contour | Existing continuous person contours and stable person IDs. Connect `frames` into the block and `tracks` into Output. |
| Person tracking → Bbox | Bounding boxes and local track IDs from the detector service. Configure its full `/v1/labels` URL in this block. |
| Face recognition | Existing face recognition, identity and stranger-confirmation component; uses the shared gallery and configured face engine. One instance per workflow. |
| Event recognition → VLM semantic recognition | Existing event engine; events and sensitivity choices come from the server catalog. Uses the shared analysis service. |
| Event recognition → System 1 recognition | Native scene classification through the configured System 1 service base URL. Uses the existing event definitions and sensitivity policies. |
| Event recognition → Neural network recognition | Fixed image/text similarity classifier through its own service base URL. Supports the six built-in safety events; no sensitivity controls or custom-event prompts. |
| Every-minute summary | Independent latest-snapshot model request; configurable 1–3600 second interval and an instruction of at most 4000 characters. Its sidebar also configures a shared OpenAI-compatible summary API. Multiple independent instances are supported. |
| Output | Publishes connected results and composes a video preview. Toggle person geometry, event alerts and summary text; choose any corner for alerts and summaries. Multiple summaries stack in their chosen corner. |

Each engine has its own draggable block and saved configuration. Keep alternatives
on the canvas, including their connections, and enable one tracking engine and
one event engine at a time. Enabling a block turns the other engines in its category
off. Apply rejects graphs with two enabled engines in one category. Group and
camera switches can disable the selected engine; they cannot enable an alternative
that is disabled in the workflow. Legacy graphs use Contour and VLM by default.

### Summary API

Select a summary block in the workflow canvas, then use **Summary service** in
its sidebar. **Use the analysis service** preserves the existing shared model
route. To use another provider, choose **OpenAI-compatible API**, enter the full
Chat Completions URL (including any custom path), the vision model ID and its
optional API key, then click **Save summary service**. Stop all analysis before
saving; the next single-frame test or analysis session uses the saved connection.
This setting is shared by all summary blocks and leaves event analysis unchanged.

The provider must accept non-streaming OpenAI-format messages with Base64 JPEG
`image_url` inputs and return `choices[].message.content`. The summary prompt
requires a JSON object containing Chinese `zh` and English `en` strings. An
interface with only text input or a different response protocol needs an adapter.

API keys are write-only: leave the key field blank to retain it, or select the
clear-key checkbox for an unauthenticated service. Changing the URL clears the
old key unless a new key is entered. The connection is atomically saved in
`summary-service.json` beside deployment settings with owner-only permissions;
camera workers sharing that data directory read it on the next request setup.
Separate fleet workers receive the connection before analysis and single-frame
tests, including an explicit empty key when authentication is cleared.
Credentials are never stored in workflow graphs or returned by the settings API.
Custom summary connections never inherit cloud or local-model environment keys.

**Request parameters (JSON)** accepts provider generation options. The default
token budget is 512; `max_tokens` or `max_completion_tokens` can set a budget up
to 4096. Set `temperature` to `null` to omit it for models that do not support
that parameter. Model, messages and streaming cannot be overridden here.

`GET /v1/deployment/summary-service` returns the connection and an
`api_key_present` flag. `PUT` accepts `mode` (`inherit` or `custom`), `endpoint`,
`model`, `request_parameters`, optional `api_key` and `clear_api_key`.

### Output appearance

Connections determine which results are available to Output. Its appearance
controls determine which of those results appear on the picture; hiding a layer
does not stop its analysis or remove its JSON results. Unconnected layers have
disabled controls. Person geometry is green; detected events such as **Smoking**
appear in a red panel; summaries appear in a separate dark panel. Negative and
failed event results do not produce an alert.

Apply the workflow and start analysis, then select Output and choose a camera to
see its applied picture. The camera selector remains usable while analysis locks
editing. Changes to appearance follow the same Apply/Stop rules as other settings.
The output JPEG API below supplies the same picture to other consumers; the
existing raw video and evidence APIs retain their contracts.

The preview is bounded to 1280 pixels on its longest side and refreshes at most
about three times per second. Person geometry uses a matching frame from the
source's bounded display history, at most two seconds old; if none remains, the
preview shows the latest source frame without mismatched geometry. Event alerts
expire 15 source seconds after their observation. Summaries expire after twice
their interval. Text wraps, long panels truncate, and stale sessions cannot supply
a preview after Stop, rebinding or a new run.

### Model services and readiness

The workbench and renderer use the normal Harness dependencies; they do not load
GPU weights. Run each model service in a separate environment and enter an address
reachable **from the camera's Harness service**. `127.0.0.1` means that service's
machine/container. The new native services bind to loopback by default; use a
private tunnel or authenticated proxy for another host. Addresses cannot contain
credentials, query strings or fragments.

| Module | Service / dependencies | Address example |
| --- | --- | --- |
| Contour | Existing configured tracking deployment; no migration required. | Shared tracking sidebar |
| Bbox | `workers/detector_rich_label`, CUDA PyTorch, Ultralytics and its weights. Worker image uses `DETECTOR_MODEL=yolo26-s`, `DETECTOR_DEVICE=cuda`, weights under `/models`; port 8092. | `http://127.0.0.1:8092/v1/labels` |
| VLM semantic recognition | Existing shared analysis service and its credentials. | Shared analysis sidebar |
| System 1 recognition | Existing VideoJev service and native CUDA dependencies; see [environment setup](videojev-candidate.md). | `http://127.0.0.1:8093` |
| Neural network recognition | New `goodvision-neural` service using CLIP ViT-L/14. Install `.[neural]` in a separate CUDA environment; requires model download/cache. | `http://127.0.0.1:8094` |

For the neural service, after installing the CUDA build of PyTorch compatible with
the GPU host and the pinned optional dependencies:

```sh
python -m pip install '.[neural]'
goodvision-neural --cache-dir /path/to/persistent/model-cache --port 8094
```

This classifier compares fixed positive/negative visual prompts on every supplied
image (one to four per window), retaining the strongest evidence for each event.
The pinned CLIP processor resizes and center-crops each decoded image. Images
that exceed the 16-million-pixel budget before or after its proportional resize
are rejected before preprocessing. Similarity
scores are not calibrated confidence and are not returned as event confidence.
It does not reason about temporal motion or apply VLM instruction policies.
The existing event confirmation pipeline still consumes its observations.
GPU accuracy, throughput and capacity require validation on the target deployment.

Optional [frozen CLIP thresholds](neural-thresholds.md) require a versioned,
externally reviewed validation artifact supplied through `--thresholds` or
`NEURAL_THRESHOLDS_FILE`. Without one, the original prompts and pair-probability
choices remain in use. No production threshold values are bundled.

Each Bbox, System 1 and neural block selects its own execution location: local,
independent server or cloud server. Local mode reuses the repository's existing
workers and requires no URL. Server and cloud modes keep separate service URLs;
the selected remote address is required before Apply. Existing workflows with a
service URL retain their server route. See [module deployment](workflow-module-deployment.md)
for local dependencies, on-demand startup and shared Docker workers. Test checks
the chosen service's readiness and reports that live analysis is required; it
does not claim that a health check proves detection accuracy. Start repeats the
readiness check. Summaries always retain the shared VLM route and credentials,
even when a different event recognition engine is selected. Switching engines or
unassigning the workflow restores the deployment settings without persisting model
overrides into those settings.

The Contour block shows its configuration separately from deployment capability.
An unconfigured tracking service does not prevent authoring or applying a graph,
but enabled tracking cannot be tested or started until its service is configured
in the tracking block's sidebar. A configured service is checked for readiness when
analysis starts. A one-frame test cannot validate continuous contours; live
analysis is required. Tracking can associate identities and located events with
people when the corresponding face/event branches are also enabled.

Model routes and credentials remain in Deployment & settings. Service connections
and sampling are configured in the corresponding workflow node sidebar.
Camera URLs and layouts are configured per source. Identity
association, evidence and delivery use the existing native
components. The canvas does not introduce a separate model registry or
nested workflow editor. Existing result/video/evidence endpoints remain usable;
workflow output connections select the new workflow result envelopes.

## Delete and restore

Manage provides deletion and Trash for groups and saved workflows. Deleting a
workflow in use requires an explicit replacement or unassignment. A replacement
immediately applies its configuration to all affected groups after the cameras
stop. Restore does
not overwrite newer group assignments. Deletion never deletes camera sources,
saved manual detection settings, gallery entries or evidence.

Removing a camera from a group or unassigning its workflow restores its existing
manual detection settings on the next Start. To exclude a connected camera from
analysis, keep it assigned and switch it off instead. Managed cameras show their
group in the live workspace and use the canvas for detection choices.

## Storage and API

The primary owns one versioned, atomic `workflow-assets.json` document beside
the deployment settings file. Without a deployment store it defaults to
`.goodvision/workflow-assets.json`; `HARNESS_WORKFLOW_SETTINGS_PATH` can override
the path. Keep this private runtime file in the service's persistent volume.
Credentials and camera URLs remain in their existing stores. Corrupt or
unwritable workflow state fails closed. Revision checks reject stale browser
writes rather than overwriting another editor.

| Method and path | Contract |
| --- | --- |
| `GET /v1/workflows` | `{document, cameras, event_catalog, tracking}`; no-store. |
| `POST /v1/workflows/command` | `{revision, action, ...fields}`; returns refreshed state. |
| `POST /v1/workflows/test` | `{graph, camera_id}`; one-frame test with source-qualified results. |
| `GET /v1/workflow-modules/{engine}/local` | Local worker state and model readiness; `engine` is `bbox`, `system1` or `neural`. |
| `POST /v1/workflow-modules/{engine}/local` | Empty object; start or check the fixed local worker while the fleet is stopped. |
| `GET /v1/fleet/cameras/{id}/workflow/results` | Applied output bindings with current-run values. |
| `GET /v1/fleet/cameras/{id}/workflow/summaries?limit=100` | `{camera_id, status, items}`; optional `node_id`; limit 1–100. |
| `GET /v1/fleet/cameras/{id}/workflow/output.jpg?node_id={output-id}&language=zh` | JPEG composed with this applied output's connected results and appearance. `language=en` selects English summary text. Returns 404 when no active output/frame is available. |
| `GET /v1/fleet/cameras/{id}/video/source/latest.jpg` | Decoded source JPEG, including when only summaries are running. |

Commands are `create_workflow {name,graph?}`,
`create_group {name,camera_ids?}`, `update_group {id,...changes}`,
`apply {workflow_id,graph?,group_ids}`, `delete_group {id}`,
`delete_workflow {id,replacement_id}` or `{id,unassign:true}`, and `restore {id}`.
The console sends the current `graph` with Apply. With a graph, `group_ids` can
be empty and the server always includes every group already assigned to that
workflow. The graph and all affected applied copies commit under one revision.
The legacy `save_workflow {id,name,graph}` API has the same immediate propagation
semantics for assigned workflows; it is not a separate console action.

Group changes can include `name`, `camera_ids`, `enabled`, `block_enabled` and
`camera_overrides`. A camera override contains nullable `enabled` and a
`block_enabled` map; null means inherit.

Graphs contain `nodes` with `id,type,x,y,enabled,config` and `edges` with
`id,from_node,from_port,to_node,to_port`. Blocks expose the typed ports shown in
the UI. The server validates types, configuration, references and connected
source-to-output paths. Invalid input returns 400, stale revisions or busy
targets 409, oversized requests 413, and unavailable storage 503. Internal
`/v1/analysis/workflow` and `/v1/analysis/workflow/test` routes are coordinated
worker controls; use the primary's public workflow API.

### Dynamic video sources in Docker

The main container enables on-demand video workers by default. Adding a Source
uses an existing free channel or `POST /v1/fleet/cameras` to allocate one. Each
additional worker runs inside the same container on a private loopback port;
there are no new Docker containers, published ports, or Docker socket mounts.
The worker count has no configured product ceiling. Actual concurrent throughput
is limited by the container resources and the services used by each workflow.

`video-sources.json` and `cameras/cameraNN/` are stored beside the existing
persistent deployment settings. Camera IDs, URLs and per-camera settings survive
container replacement and restart. Existing launcher-provisioned peers retain
their original destinations. New cameras start disconnected with analysis stopped.
Removing a Source from a workflow leaves its saved channel available for reuse.
Worker failures are reported and the next request restarts the affected process.
Increasing the provisioned capacity does not invalidate an existing source registry.
Workers refresh the shared people library once per second after face recognition
has initialized. Enrollment and deletion update their recognition snapshots and
invalidate obsolete identity/display results without restarting the workers.

Video connection, Summary, face and Bbox workflows do not reserve Contour lanes.
Contour admission still checks the actual tracking server capacity when enabled.
`HARNESS_FLEET_CAPACITY` remains compatible with old installations, but no longer
needs to be increased to add video sources. `HARNESS_FLEET_DYNAMIC=false` is an
optional compatibility switch for administrators who require fixed topology.

Dynamic workers require POSIX socket inheritance. Linux Docker containers support
this on Linux hosts and on Windows Docker Desktop in Linux-container mode.
Running the Python application directly on native Windows retains the existing
fixed, provisioned camera channels; dynamic allocation is unavailable there,
even if `HARNESS_FLEET_DYNAMIC=true` is set.
