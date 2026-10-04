# Updating running Docker deployments

[Repository overview](../../README.md) · [Tools](../../tools/README.md)

`tools/update_docker.py` updates an existing Docker Compose deployment of the
main harness or GoodVision Contour. It uses Python 3.10 or later with the standard
library, Docker Compose v2 or later and local Linux containers. Run it on the host where
the selected component is installed; for separate main and GPU hosts, run the
matching component on each host.

The script discovers the installed Compose project, working directory and exact
configuration files from container labels. It preserves that deployment's service
names, ports, mounts and configuration. It updates all `harness` / `harness-N`
lanes together, or the GPU Contour `contour` and `engine` pair together. Scaled
replicas of one Compose service are unsupported; use separate `harness-N` lanes.
It does not update other workers or install a new deployment.

## Before an update

Stop every analysis using the selected deployment and wait for cleanup to finish.
For the harness, also disable automatic analysis startup in its saved deployment
settings through the console. The saved `autostart` setting must be false; the
original `.env` can retain its startup default. Selected containers must still be running so the
updater can inspect their status. Stop analysis on any other harness host using a
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

## Build from updated source

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

## Use prepared images, including offline installations

For customer deployments with images and no buildable source, prepare images on
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

## What happens during an update

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

## Keep the active image selection on later restarts

After either an update or rollback, append `active.compose.json` **last** whenever
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
Existing deployment launchers do not automatically append this overlay. Omitting
it can recreate services with older tags from the original Compose files. The
updater itself includes the active overlay when replacing services; subsequent
source builds use the original build definitions.

## Roll back the last update

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
