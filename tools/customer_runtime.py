"""Replace customer images using the installed Docker runtime configuration."""

from __future__ import annotations

import hashlib
import json
import uuid

if __package__:
    from .harness_update.apply_update import replacement_config
    from .update_docker import (
        LABEL,
        UpdateError,
        assert_idle,
        current_containers,
        image_id,
        launcher_references,
        tag_launcher_images,
        wait_ready,
        write_json,
    )
else:
    from harness_update.apply_update import replacement_config
    from update_docker import (
        LABEL,
        UpdateError,
        assert_idle,
        current_containers,
        image_id,
        launcher_references,
        tag_launcher_images,
        wait_ready,
        write_json,
    )


def runtime_fingerprint(container):
    # Docker-assigned IDs and addresses are excluded by replacement_config.
    # Only the digest is recorded; runtime settings and keys stay in memory.
    config = replacement_config(container, container["Image"])
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def runtime_update(docker, deployment, images, *, finish=None, timeout=900):
    """Caller owns update_lock and drains local analysis before calling this."""
    originals = current_containers(docker, deployment)
    if {s: c["Id"] for s, c in originals.items()} != {
        s: c["Id"] for s, c in deployment.containers.items()
    }:
        raise UpdateError("Installed containers changed while preparing the update.")
    if set(images) != set(originals):
        raise UpdateError("The update must include every selected service.")
    references = launcher_references(docker, deployment, images, deduplicate=False)
    reference_ids = {
        s: image_id(docker, reference) for s, reference in references.items()
    }
    accepting = assert_idle(docker, deployment)
    before = {s: c["Image"] for s, c in originals.items()}
    if before == images:
        try:
            tag_launcher_images(docker, references, images)
            if finish:
                finish()
        except (Exception, KeyboardInterrupt) as error:
            try:
                tag_launcher_images(docker, references, reference_ids)
            except UpdateError as recovery_error:
                raise UpdateError(
                    "Recovery did not become ready. Keep the saved state "
                    "and contact support."
                ) from recovery_error
            raise UpdateError(
                "Update failed; previous images were restored."
            ) from error
        return

    transaction = uuid.uuid4().hex
    prepared = {}
    record_path = deployment.state_dir / "runtime-latest.json"
    retained_backups = []
    if record_path.is_file():
        previous = json.loads(record_path.read_text("utf-8"))
        if previous.get("project") == deployment.project:
            known = set(previous.get("retained_backups", []))
            known.update(
                item["old_id"] for item in previous.get("containers", {}).values()
            )
            for identity in known:
                try:
                    old = docker.json("inspect", identity)[0]
                except (UpdateError, IndexError):
                    continue
                labels = old["Config"].get("Labels") or {}
                if (
                    not old["State"].get("Running")
                    and "-before-" in old["Name"]
                    and (
                        (
                            labels.get(LABEL + "project") == deployment.project
                            and labels.get(LABEL + "service") in deployment.services
                        )
                        or (
                            deployment.manual_names
                            and any(
                                old["Name"].lstrip("/").startswith(name + "-before-")
                                for name in deployment.manual_names.values()
                            )
                        )
                    )
                ):
                    retained_backups.append(identity)
    record = {
        "transaction": transaction,
        "component": deployment.component,
        "project": deployment.project,
        "before": before,
        "after": images,
        "references": references,
        "accepting_sessions": accepting,
        "status": "preparing",
        "containers": {},
        "retained_backups": retained_backups,
    }
    fingerprints = {s: runtime_fingerprint(c) for s, c in originals.items()}
    # Validate every runtime before creating any container. Static IPs and legacy
    # links need a migration; the existing verified updater refuses them too.
    configs = {s: replacement_config(c, images[s]) for s, c in originals.items()}
    helper_image = (
        next(iter(images.values()))
        if deployment.component == "harness"
        else images["contour"]
    )
    names = {s: c["Name"].lstrip("/") for s, c in originals.items()}

    def remove_prepared():
        errors = []
        for identity in prepared.values():
            try:
                docker.run("rm", "-f", identity)
            except UpdateError as error:
                errors.append(error)
        if errors:
            raise errors[0]

    try:
        for service, config in configs.items():
            labels = config["Labels"] = config.get("Labels") or {}
            labels[LABEL + "image"] = images[service]
            labels["goodvision.update.transaction"] = transaction
            labels["goodvision.update.original_id"] = originals[service]["Id"]
            labels["goodvision.update.launcher_image"] = references[service]
            if deployment.manual_names:
                labels["goodvision.update.manual_project"] = deployment.project
                labels["goodvision.update.manual_role"] = service
            name = names[service] + "-before-staging-" + transaction
            try:
                identity = docker.create_container(config, name, helper_image)
            except Exception:
                # A lost helper response can follow a successful create. Remove
                # only a container carrying this transaction's exact identity.
                try:
                    value = docker.json("inspect", name)[0]
                    if (
                        value["Config"]["Labels"].get("goodvision.update.transaction")
                        == transaction
                    ):
                        prepared[service] = value["Id"]
                except (UpdateError, IndexError):
                    pass
                raise
            prepared[service] = identity
            record["containers"][service] = {
                "name": names[service],
                "old_id": originals[service]["Id"],
                "new_id": identity,
                "configuration": fingerprints[service],
            }
        fresh = current_containers(docker, deployment)
        if any(
            fresh[s]["Id"] != originals[s]["Id"]
            or runtime_fingerprint(fresh[s]) != fingerprints[s]
            for s in originals
        ):
            raise UpdateError("Installed settings changed while preparing the update.")
        assert_idle(docker, deployment)
        record["status"] = "prepared"
        write_json(record_path, record)
    except (Exception, KeyboardInterrupt):
        remove_prepared()
        raise

    paused = False
    switched = False
    try:
        if deployment.component == "contour":
            paused = True
            docker.api(
                originals["contour"]["Id"],
                "contour",
                "/api/admission",
                {"accepting_sessions": False},
            )
            assert_idle(docker, deployment)
        for service, identity in before.items():
            docker.run(
                "image",
                "tag",
                identity,
                f"goodvision-update-backup/{deployment.project}-{service}:{transaction}",
            )
        try:
            switched = True
            # Stop every old GPU workload before starting any replacement.
            docker.run(
                "stop",
                "--time",
                "30",
                *[c["Id"] for c in originals.values()],
                timeout=180,
            )
            for service, original in originals.items():
                docker.run(
                    "rename", original["Id"], names[service] + "-before-" + transaction
                )
                docker.run("rename", prepared[service], names[service])
            docker.run("start", *prepared.values(), timeout=180)
            wait_ready(docker, deployment, timeout)
            tag_launcher_images(docker, references, images)
            record["status"] = "ready"
            write_json(record_path, record)
            # Make launcher metadata the last fallible commit operation, so a
            # state-file failure cannot roll images back after metadata changed.
            if finish:
                finish()
        except (Exception, KeyboardInterrupt) as error:
            try:
                remove_prepared()
                for service, original in originals.items():
                    current = docker.json("inspect", original["Id"])[0]
                    if current["Name"].lstrip("/") != names[service]:
                        docker.run("rename", original["Id"], names[service])
                docker.run("start", *[c["Id"] for c in originals.values()], timeout=180)
                wait_ready(docker, deployment, timeout)
                tag_launcher_images(docker, references, reference_ids)
                record["status"] = "restored"
                write_json(record_path, record)
            except (Exception, KeyboardInterrupt) as recovery_error:
                raise UpdateError(
                    "Recovery did not become ready. Keep the saved state "
                    "and contact support."
                ) from recovery_error
            raise UpdateError(
                "Update failed; previous images were restored."
            ) from error
    finally:
        try:
            if not switched:
                remove_prepared()
        finally:
            if paused:
                import sys

                failure_pending = sys.exc_info()[0] is not None
                try:
                    gateway = current_containers(docker, deployment)["contour"]
                    docker.api(
                        gateway["Id"],
                        "contour",
                        "/api/admission",
                        {"accepting_sessions": accepting},
                    )
                except UpdateError as error:
                    if not failure_pending:
                        raise UpdateError(
                            "Could not restore Contour admission; "
                            "check the node console."
                        ) from error

    # Keep original containers through the launcher metadata transaction. Once
    # committed, remove stopped backups without deleting their volumes.
    retained = list(retained_backups)
    for original in originals.values():
        try:
            docker.run("rm", original["Id"])
        except UpdateError:
            retained.append(original["Id"])
    if retained:
        record["retained_backups"] = retained
        try:
            write_json(record_path, record)
        except OSError:
            # The ready record already contains these old IDs. Future updates
            # recheck them, even if this optional cleanup annotation cannot save.
            pass
