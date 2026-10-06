# Updating running Docker deployments

[Repository overview](../../README.md) · [Tools](../../tools/README.md)

## Customer updates on Windows

Keep Docker Desktop running, extract the supplied Windows update ZIP completely,
and double-click **`Update.exe`**; no Git, Python or terminal commands are needed.
The window finds the installed component (asks you to choose if several
installations exist), stops local harness analysis, installs the update and shows
progress and the result; it restores your saved autostart setting afterward.

If automatic detection finds nothing, paste the exact container or image name
shown in Docker Desktop into **Container or image name**, then select
**Find installation**. The updater checks that the matching container is running
GoodVision before offering it. If an image matches several installations, choose
the installation to update from the list. A match in a recognized installation
updates all its main lanes. Otherwise, choosing a main container updates only
that container. **Retry** checks the name again before trying another update.

Your models, data and configuration are retained, and you can continue opening
GoodVision as usual. For Contour on another
computer, stop analysis in the main console first and run the Contour update
package on that computer.

## Maintainer and source-deployment tools

The commands below are for maintainers and source deployments. Customers use
the packaged Windows updater above.

### Package a Windows customer update

Prepare and validate the candidate **Linux/amd64** images locally before
packaging. On a Windows build machine with Python 3.10 or later, install
PyInstaller in the maintainer environment and package the selected component:

```powershell
python -m pip install pyinstaller
python tools/package_customer_docker_update.py --output dist/goodvision-harness-update-next.zip --release next --component harness --image harness=goodvision-harness:next
python tools/package_customer_docker_update.py --output dist/goodvision-contour-update-next.zip --release next --component contour --image contour=goodvision-contour:next --image engine=goodvision-contour-runtime:next
```

Replace `next` with a new release label and use the corresponding validated
image tags. The packager builds a standalone Windows `Update.exe`, exports the
selected images, and creates a ZIP containing the executable, `images.tar.gz`,
`update-package.json` and short customer instructions. It also writes a ZIP
checksum alongside the output. It never overwrites an existing output ZIP.
Packaging does not build product images or include customer models, data or keys.

The harness package contains one `harness` image that the window applies to every
lane in a recognized project, whether discovered automatically or selected by
name. A main container without recognized workload labels is updated individually.
The Contour package requires both `contour` and `engine`
images. Deliver the matching package to each host; customers extract it and
double-click the executable. Neither Python nor PyInstaller is required on the
customer machine. To reuse a previously built Windows updater, pass
`--executable /path/to/Update.exe`; this also permits packaging on another build
host without rebuilding the executable.

Customer updates use inspected runtime snapshots, preserving `Config`,
`HostConfig`, networks, custom container names and original image tags. This keeps
existing environment values, port bindings, mounts and network aliases when an
older updater changed the running container without changing its Compose files.
Project/service-labeled historical Docker runs are supported even when their
original Compose files are unavailable. The source CLI below still uses Compose.

Original launchers reopen the updated version automatically. The customer updater
refuses image tags shared by another installation. For legacy harness
installations with `RELEASE.json`, the candidate
must retain every file in the original source inventory; launcher metadata is
verified and refreshed from the actual candidate source hashes. Packages for an
incompatible edition or with missing inventory files are refused.

### Historical compatibility evidence and limits

| Recorded release or workflow | Git evidence | Compatibility requirement |
| --- | --- | --- |
| `20260920-diagnostics2` source update | `f396940` introduced `apply_update.py` and its inspected-runtime replacement. | Preserve installed container configuration and Windows launcher metadata. |
| `20260921-realtime1` and subsequent source-update chains | `eb4cb92` added `continuous_playback_vlm`; `d68b654` added verified chain handling. | Preserve the injected `LOCAL_CAMERA_TARGET_FPS=15`, `SCENE_ENGINE_MODE=active` and `SCENE_INDEPENDENT_VLM=true` values even when the original Compose hash remains unchanged. |
| `20260923-image1` / `20260923-full1` image update | `ccff052` added `package_image_update.py`, using the same runtime replacement and realtime flag. | Image updates must retain the installed runtime rather than recreate it from an older Compose definition. |
| Full.1 / Full.2 / Full.3 console APIs | `fa145ce`, `c805c3b` and `098fd84` have deployment settings and `/v1/sessions/stop`; `6e80baf` later adds `/v1/analysis/stop`. | Preserve saved autostart and use the retained legacy stop endpoint when the newer endpoint returns 404. |
| Standalone GPU Contour | `f5b651f` introduced the `contour` / `engine` service pair. | Update both services using their workload labels and preserve their model mounts and node key. |
| `20260920-memory1` Contour update ZIP | Available `dist` package and archived deployment records. | Discover project `goodvision-contour` and retain `goodvision-contour:0.1.0` / `goodvision-contour-runtime:0.1.0` launcher tags. |
| `Contour-20260926.zip` | Archived Compose and installation scripts. | Discover project `contour`, with `goodvision-contour:20260926` / `goodvision-contour-engine:20260926`; no fixed project or engine-image name. |
| `20261001-update1` validation | Archived `validate_update.py` / release records. | Support custom main-container names, multiple lanes, and Docker runs carrying project/service labels without Compose paths. |

Automatic discovery uses workload project and service labels. When they are
missing, the manual field accepts an exact container name or an existing image
name and verifies candidates through their application APIs. Image matches with
multiple verified installations require an explicit choice. A match with
recognized project/service labels selects the complete installation, including
every main lane; an unlabeled or unrecognized main role selects only that verified
container. Contour must resolve a verified gateway and engine pair. A publisher
or helper is not accepted just because it uses the
application image. Static container IPs and legacy Docker links are refused by
the existing replacement policy. Edition and filesystem-integrity checks remain
in force; missing inventory entries require a separately reviewed migration.
Known Windows base images require their original verified launcher metadata.
Label-only installations share a project lock in the user's local application
data across extracted update folders.

The archived `20260930-legacy-gpuhost1` source inventory includes
`analysis_gateway.html`, `analysis_gateway.py`, `gateway_video.py` and
`video_relay.py`, which are absent from the current tracked main source. Where
the original launcher inventory lists them, an update image must retain those
files or supply an approved edition migration. A current-source-only image is
refused for that inventory; the updater never removes entries to pass checks.

This audit inspected tracked source and release records. It did not run upgrades
against real historical images; those require a working Docker daemon and the
corresponding archives. Tracked records do not map each accepted `20260919` image
ID to an exact source commit.

### Source-deployment updater

`tools/update_docker.py` updates an existing Docker Compose deployment of the
main harness or GoodVision Contour. It uses Python 3.10 or later with the standard
library, Docker Compose v2 or later and local Linux containers. Run it on the
host where the selected component is installed; for separate main and GPU hosts, run the
matching component on each host.

The script discovers the installed Compose project, working directory and exact
configuration files from container labels. It preserves that deployment's service
names, ports, mounts and configuration. It updates all `harness` / `harness-N`
lanes together, or the GPU Contour `contour` and `engine` pair together. Scaled
replicas of one Compose service are unsupported; use separate `harness-N` lanes.
It does not update other workers or install a new deployment.

### Before a command-line update

Stop every analysis using the selected deployment and wait for cleanup to finish.
For the harness, also disable automatic analysis startup in its saved deployment
settings through the console. The saved `autostart` setting must be false; the
original `.env` can retain its startup default. Selected containers must still be
running so the updater can inspect their status. Stop analysis on any other harness host using a
Contour node before updating that node; no Contour sessions or requests may remain.

Keep the installed Compose files, `.env` and any explicitly selected environment
files unchanged. The updater checks the resolved configuration against the live
installation and refuses configuration drift. Source files in the installed
build context may be updated. Configuration, model or schema changes need their
own deployment procedure.

Use the same custom environment files as the original deployment, with repeated
`--env-file /absolute/path/to/file` arguments in the same order when discovery
cannot recover them. Without overrides, the updater uses recorded environment
file paths from Compose labels when available, or the normal project `.env`.

### Build from updated source

Update the source in the **installed build context**, which may differ from the
directory containing the updater. For a deployment checkout already on `main`,
run from the repository root:

```sh
git pull --ff-only origin main
python3 tools/update_docker.py harness --check-only
python3 tools/update_docker.py harness
```

On the Contour host, use:

```sh
git pull --ff-only origin main
python3 tools/update_docker.py contour --check-only
python3 tools/update_docker.py contour
```

On Windows with Docker Desktop in Linux-container mode, use `python` in
PowerShell from the deployment repository root:

```powershell
git pull --ff-only origin main
python tools/update_docker.py harness --check-only
python tools/update_docker.py harness
```

The updater never pulls Git changes. Its default mode rebuilds the selected
services using the installed Compose configuration and build paths. Dependencies
or base images may need network access; this is a full image build, not the
Windows package updater's offline code-only workflow. It does not download models.

`--check-only` validates discovery, configuration and idle status without
building, recreating containers, pausing admission or writing update state.
If more than one eligible project is installed, specify `--project` explicitly:

```sh
python3 tools/update_docker.py harness --project goodvision-harness --check-only
python3 tools/update_docker.py contour --project goodvision-contour --timeout 1800
```

`--timeout` is the readiness-wait limit and defaults to 900 seconds. Allow enough
time for Contour model loading on the target host. The build has a separate
two-hour limit. See `--help` for all options.

### Use prepared images for a command-line update

For image-only deployments managed from the command line, prepare images on
a suitable build host, transfer them, and load them on the deployment host first:

```sh
docker load -i /path/to/goodvision-update-images.tar.gz
python3 /path/to/update_docker.py harness --image harness=goodvision-harness:next --check-only
python3 /path/to/update_docker.py harness --image harness=goodvision-harness:next
```

When using `--image`, provide one `SERVICE=IMAGE` argument for **every selected
service**. Use the installed Compose service names, not container names. A
two-lane harness deployment needs both `--image harness=...` and
`--image harness-02=...`. Contour needs both images:

```sh
python3 /path/to/update_docker.py contour \
  --image contour=goodvision-contour:next \
  --image engine=goodvision-contour-runtime:next --check-only
python3 /path/to/update_docker.py contour \
  --image contour=goodvision-contour:next \
  --image engine=goodvision-contour-runtime:next
```

In Windows PowerShell, put both image arguments on one line:

```powershell
docker load -i "D:\Contour\goodvision-update-images.tar.gz"
python tools/update_docker.py contour --image contour=goodvision-contour:next --image engine=goodvision-contour-runtime:next --check-only
python tools/update_docker.py contour --image contour=goodvision-contour:next --image engine=goodvision-contour-runtime:next
```

All supplied images must already exist locally and match the deployment's
architecture, GPU runtime and application expectations. This mode does not build
or pull them. Reusing a tag is supported because the updater records immutable
image IDs; distinct release tags make operations easier to audit.

### What happens during a command-line update

The script validates the installation and prepares all candidate images before
recreating services. For Contour it then pauses new-session admission, confirms
that no sessions or requests remain, and remembers the previous admission state.
It pins candidate image IDs, recreates only the selected services, and waits for
application readiness, including Contour model readiness. Admission returns to
its previous setting; harness analysis remains stopped until an operator starts it.

A failed replacement attempts to restore the previous images and Contour
admission state, then reports failure. Check the reported recovery result before
resuming analysis. The script does not prune images or volumes. Keep the previous
images locally available for rollback.

Update metadata lives under the installed Compose working directory:

```text
.goodvision/docker-updates/<project>/latest.json
.goodvision/docker-updates/<project>/active.compose.json
```

These ignored files contain image IDs, a configuration fingerprint, transaction
status and the previous Contour admission flag. They do not store environment
values, credentials or copies of deployment settings. Existing `.env`, model
directories, mounted configuration and persistent data are not rewritten.
Repeated updates are supported; only the most recent update **per Compose
project** is retained for rollback.

### Keep the command-line image selection on later restarts

This section applies to command-line updates. The customer executable keeps
original launchers working automatically.

After either a command-line update or rollback, append `active.compose.json` **last** whenever
using Compose to recreate services. Use the original project, working directory,
Compose files and environment files, followed by that overlay. For example, from
the installed Contour working directory:

```sh
docker compose -p goodvision-contour -f compose.yaml -f compose.gpu.yaml \
  -f .goodvision/docker-updates/goodvision-contour/active.compose.json \
  up -d --no-build --pull never
```

For a harness deployment using only `compose.yaml`, from its installed working
directory:

```sh
docker compose -p goodvision-harness -f compose.yaml \
  -f .goodvision/docker-updates/goodvision-harness/active.compose.json \
  up -d --no-build --pull never
```

Adapt those examples to the discovered project and original configuration files.
After command-line updates, deployment launchers do not automatically append this overlay. Omitting
it can recreate services with older tags from the original Compose files. The
updater itself includes the active overlay when replacing services; subsequent
source builds use the original build definitions.

### Roll back the last update

Stop analysis and turn off harness autostart before rollback too. Keep the
deployment's Compose files and environment files exactly as they were during the
update; rollback verifies their fingerprints and requires the saved old images.

```sh
python3 tools/update_docker.py harness --rollback
python3 tools/update_docker.py contour --rollback --project goodvision-contour
```

Rollback also accepts the original `--env-file` arguments. It can recover stopped
services when a previous transaction record exists. It replaces images and updates
the persistent overlay to the previous image IDs. It does not roll back data,
database/schema changes, models or mounted configuration. Test changes that affect
persistent data before deployment; an image rollback cannot reverse them.

The existing [Windows offline harness update package](../../tools/harness_update/README.md)
and Contour initial deployment scripts remain available with their own procedures.
