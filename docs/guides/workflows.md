# Workflow canvas

Open **Workflows** in the console. The canvas fits the browser window; drag empty
space to pan, use the zoom controls or Ctrl/Command + wheel, and choose **Fit** to
see the graph. Existing live video, people library, event replay, output and
deployment pages remain available without reconnecting cameras.

## Build, test and apply

1. Create a workflow. Add blocks by clicking the library or dragging them onto the
   canvas. Connect an output port to an input of the same type. Select a block to
   configure it. Delete removes a block and its connections; Undo restores them.
2. Connect **Workflow input → Face recognition → Output**. Add a parallel
   **Workflow input → Every-minute summary → Output** path. Each summary instance
   has its own interval and instruction; the default is 60 seconds.
3. **Save draft** persists the recipe without changing any camera's applied copy.
   Incomplete drafts can be saved. Apply and Test require valid connected paths.
4. **Test** uses one current frame from a connected, stopped camera. It runs the
   actual configured face/event/summary adapters without publishing events,
   sending webhooks or changing the applied setup. The result identifies its
   camera and source frame. Dry-run results are explicitly synthetic. A single
   snapshot does not test temporal confirmation, tracking or event evidence.
5. Create camera groups in **Manage**, choosing already deployed camera slots.
   Each camera belongs to at most one group. Adding a slot to a group does not
   provision a service or change its saved video address. Connect video in the
   live workspace using the existing network and gateway controls.
6. **Apply to groups…** copies the saved graph to the selected groups. Other
   groups keep their applied copies. All affected cameras must be stopped and
   finished cleaning up. An invalid or unavailable target rejects the whole
   operation. Apply never starts analysis or disconnects a preview.
7. Use **Start all analysis** in the live workspace. The coordinator installs the
   applied plans and preflights the connected cohort before starting it. Cameras
   whose applied pipelines are all disabled are excluded. **Stop all analysis**
   stops processing while retaining video connections.

## Camera and group switches

Camera groups display the applied graph read-only. Select **All cameras** for
group switches, or one camera for its `Inherit`, `On` and `Off` overrides. A group
or group block switched off is a hard gate: a camera's `On` cannot bypass it.
Changes require the affected cameras to be stopped. Draft editing remains
available during analysis. Execution still uses the existing shared Start/Stop
controls; this release does not hot-swap running graphs.

The live results pane displays camera-qualified outputs and recent summary
history. A summary uses the latest source snapshot at each interval, starting
after the first interval. It does not summarize a preceding video window. Each
summary has one request in flight, a timeout and bounded history. Errors are
reported on that block; other processing continues. Summary history is held in
memory for the current run, not an archive.

## Supported blocks and settings

| Block | Configuration and behavior |
| --- | --- |
| Workflow input | The connected camera's decoded frames, shared between branches. |
| Face recognition | Existing face recognition, identity and stranger-confirmation component; uses the shared gallery and configured face engine. One instance per workflow. |
| Event recognition | Existing event engine; supported events and sensitivity choices come from the server catalog. One instance per workflow. |
| Every-minute summary | Independent latest-snapshot model request; configurable 1–86400 second interval and instruction. Multiple independent instances are supported. |
| Output | Publishes connected frames, faces, events or summaries through the workflow output API. A port accepts one connection; use separate outputs for multiple summary instances. |

Model routes, credentials, hardware, sampling and tracking options stay in the
existing deployment/configuration controls. Tracking, identity association,
evidence and delivery retain their existing processing behavior inside the
native components. The canvas does not introduce a separate model registry or
nested workflow editor. Existing result/video/evidence endpoints remain usable;
workflow output connections select the new workflow result envelopes.

## Delete and restore

Manage provides deletion and Trash for groups and saved workflows. Deleting a
workflow in use requires an explicit replacement or unassignment. Restore does
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
| `GET /v1/workflows` | `{document, cameras, event_catalog}`; no-store. |
| `POST /v1/workflows/command` | `{revision, action, ...fields}`; returns refreshed state. |
| `POST /v1/workflows/test` | `{graph, camera_id}`; one-frame test with source-qualified results. |
| `GET /v1/fleet/cameras/{id}/workflow/results` | Applied output bindings with current-run values. |
| `GET /v1/fleet/cameras/{id}/workflow/summaries?limit=100` | `{camera_id, status, items}`; optional `node_id`; limit 1–100. |
| `GET /v1/fleet/cameras/{id}/video/source/latest.jpg` | Decoded source JPEG, including when only summaries are running. |

Commands are `create_workflow {name,graph?}`, `save_workflow {id,name,graph}`,
`create_group {name,camera_ids?}`, `update_group {id,...changes}`,
`apply {workflow_id,group_ids}`, `delete_group {id}`,
`delete_workflow {id,replacement_id}` or `{id,unassign:true}`, and `restore {id}`.
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
