"""Unified, standard-library Docker installer. Plan/config never start Docker.

Each module owns a Compose project and volumes. Only an explicitly named network
is shared. No Docker socket is mounted in the application, and stop never deletes
data, models, or the shared network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

if __package__:
    from . import docker_modules as spec
else:
    import docker_modules as spec


def location():
    directory = Path(__file__).resolve().parent
    if (directory / "module-package.json").is_file():
        return directory, directory / "source"
    return directory.parent, directory.parent


def read_package(root):
    path = root / "module-package.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def verify_package(root, package):
    if package is None:
        return
    for relative, expected in package["files"].items():
        original = root / relative
        path = original.resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or original.is_symlink()
        ):
            raise ValueError("Package is incomplete or contains an invalid path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Package file has changed: {relative}")


def run(command, **options):
    # A list, never a shell expression. Secrets are read from files/environment.
    return subprocess.run(command, check=True, **options)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def configuration(args, package, modules=None):
    namespace = spec.name(args.namespace)
    state = (args.state_dir or Path.home() / ".goodvision/docker" / namespace).resolve()
    settings_path = state / "settings.json"
    incoming_version = package["version"] if package else args.version
    settings = spec.defaults(namespace, incoming_version or "dev")
    if "bbox" in (modules or []) and not settings_path.is_file():
        settings["bbox_device"] = "auto"
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if settings["namespace"] != namespace:
            raise ValueError("This data directory belongs to a different deployment")
    versions = {module: settings["version"] for module in spec.MODULES}
    versions.update(settings.get("module_versions", {}))
    fixed = {"version": package["version"]} if package else {}
    if (package and package["module"] == "bbox"
            and package["bbox_device"] != "auto"):
        fixed["bbox_device"] = package["bbox_device"]
    settings.update(fixed)
    if args.bbox_device is not None and args.bbox_device != settings["bbox_device"]:
        settings.get("module_images", {}).pop("bbox", None)
        settings.pop("bbox_runtime_device", None)
    for key in ("version", "bbox_device", "bind", "gpu_id", "contour_capacity"):
        value = getattr(args, key)
        if value is not None:
            if key in fixed and value != fixed[key]:
                raise ValueError("Package version and Bbox device cannot be changed")
            settings[key] = value
    if incoming_version is not None:
        selected = modules or args.modules or ([package["module"]] if package else [])
        for module in selected:
            if versions[module] != settings["version"]:
                settings.get("module_images", {}).pop(module, None)
            versions[module] = settings["version"]
    settings["module_versions"] = versions
    for module in spec.PORTS:
        value = getattr(args, module + "_port")
        if value is not None:
            settings["ports"][module] = value
    for module in settings["urls"]:
        value = getattr(args, module + "_url")
        if value is not None:
            settings["urls"][module] = spec.endpoint(value)
    return state, spec.validate(settings)


def compose_command(state, module, *arguments):
    return [
        "docker",
        "compose",
        "--project-directory",
        str(state),
        "-f",
        str(state / f"compose.{module}.json"),
        *arguments,
    ]


def check_contour_model_access(path):
    """Check the image's UID/GID without changing existing host permissions."""
    if os.name == "nt":
        return
    info = path.stat()
    required = 0o5 if path.is_dir() else 0o4
    if info.st_uid == 10004:
        required <<= 6
    elif info.st_gid == 10004:
        required <<= 3
    if info.st_mode & required != required:
        raise ValueError(
            f"Contour model path requires read/traverse access for UID/GID 10004: "
            f"{path}. Existing permissions were left unchanged."
        )


def initialize(state, modules):
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    for module in set(modules) & {"bbox", "system1", "contour"}:
        path = state / "models" / module
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            path.mkdir()
        except FileExistsError:
            if not path.is_dir():
                raise ValueError(f"Model path is not a directory: {path}") from None
        else:
            if module == "contour" and os.name != "nt":
                # Only this newly created mount root is made container-readable.
                # Its host-side state directory remains private; secrets stay 0700.
                path.chmod(0o755)
        if module == "contour":
            check_contour_model_access(path)
    if "contour" in modules:
        directory = state / "secrets"
        directory.mkdir(mode=0o700, exist_ok=True)
        path = directory / "contour-key"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(secrets.token_urlsafe(36) + "\n")
        if os.name != "nt":
            # The parent directory is owner-only; the mounted file must be
            # readable by the non-root Contour container even with umask 077.
            path.chmod(0o644)


def ensure_network(settings):
    network = f"goodvision-{settings['namespace']}"
    result = subprocess.run(
        ["docker", "network", "inspect", network], capture_output=True, text=True
    )
    if result.returncode:
        # A daemon outage must not be mistaken for a missing network.
        run(["docker", "info"], stdout=subprocess.DEVNULL)
        result = subprocess.run(
            [
                "docker",
                "network",
                "create",
                "--label",
                "io.goodvision.modules=true",
                network,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:  # Another module installer may have created it.
            existing = run(
                ["docker", "network", "inspect", network],
                capture_output=True,
                text=True,
            )
            if (
                json.loads(existing.stdout)[0]
                .get("Labels", {})
                .get("io.goodvision.modules")
                != "true"
            ):
                raise ValueError(f"Network {network} is owned by another application")
    else:
        value = json.loads(result.stdout)[0]
        if value.get("Labels", {}).get("io.goodvision.modules") != "true":
            raise ValueError(
                f"Network {network} exists but is not a GoodVision module network"
            )


def prepare_command(module, document, state, source):
    service = "engine" if module == "contour" else module
    image = document["services"][service]["image"]
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        "0:0",
        "--entrypoint",
        "python",
        "--mount",
        f"type=bind,source={state / 'models' / module},target=/models",
        "--env",
        "HF_HUB_OFFLINE=0",
        "--env",
        "HF_TOKEN",
    ]
    if module == "contour":
        command += [
            "--mount",
            "type=bind,source="
            + str(source / "workers/contour/download_model.py")
            + ",target=/tmp/download_model.py,readonly",
            image,
            "/tmp/download_model.py",
            "--destination",
            "/models",
        ]
    elif module == "bbox":
        command += [
            "--workdir",
            "/models",
            image,
            "-c",
            "from ultralytics import YOLO; YOLO('yolo26s.pt')",
        ]
    else:
        command += [
            image,
            "-c",
            "from huggingface_hub import snapshot_download; "
            "from goodvision_harness.videojev_core import MODEL,REVISION; "
            "snapshot_download(MODEL,revision=REVISION,"
            "cache_dir='/models/huggingface'); "
            "from ultralytics import YOLO; YOLO('/models/yolo26s.pt')",
        ]
    return command


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "action",
        choices=(
            "plan",
            "configure",
            "build",
            "prepare-models",
            "up",
            "status",
            "stop",
            "logs",
            "update",
            "rollback",
        ),
    )
    value.add_argument("--modules", nargs="+", choices=spec.MODULES)
    value.add_argument("--namespace", default="default")
    value.add_argument("--state-dir", type=Path)
    value.add_argument("--version")
    value.add_argument("--bundle", type=Path, help="Prebuilt module update manifest")
    value.add_argument("--bbox-device", choices=("auto", "cpu", "cuda"))
    value.add_argument("--platform", choices=("linux/amd64", "linux/arm64"),
                       help="Bbox cross-build target; default is the Docker engine")
    value.add_argument("--bind")
    value.add_argument("--gpu-id")
    value.add_argument("--contour-capacity", type=int)
    for module in spec.PORTS:
        value.add_argument(f"--{module}-port", type=int)
    for module in ("bbox", "system1", "neural", "contour"):
        value.add_argument(f"--{module}-url")
    value.add_argument("--build", action="store_true", help="Build before up")
    value.add_argument("--wait-timeout", type=int, default=900)
    value.add_argument(
        "--dry-run", action="store_true", help="Print only; no Docker or writes"
    )
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.dry_run or args.action in ("plan", "status", "logs"):
        return execute(args)
    try:
        namespace = spec.name(args.namespace)
        state = (args.state_dir or Path.home() / ".goodvision/docker" / namespace)
        # Acquire before configuration reads its shared settings snapshot.
        with spec.state_lock(state):
            return execute(args)
    except (ValueError, OSError) as error:
        print(f"Deployment not completed: {error}", file=sys.stderr)
        return 1


def execute(args):
    root, source = location()
    try:
        package = read_package(root)
        verify_package(root, package)
        bundle = None
        if args.action == "update":
            if args.bundle is None or args.version is not None:
                raise ValueError("Use update --bundle; the package fixes its version")
            if __package__:
                from . import module_updates
            else:
                import module_updates
            bundle = module_updates.read_bundle(args.bundle.resolve())
        modules = list(
            dict.fromkeys(
                args.modules or ([bundle["module"]] if bundle else
                                 [package["module"]] if package else ["main", "bbox"])
            )
        )
        if package and modules != [package["module"]]:
            raise ValueError("This installer contains only its named module")
        if args.wait_timeout < 1:
            raise ValueError("Wait timeout must be positive")
        if args.platform and "bbox" not in modules:
            raise ValueError("--platform currently selects the Bbox build target")
        updating = args.action in ("update", "rollback")
        if updating and args.version is not None:
            raise ValueError("Update and rollback use their recorded versions")
        state, settings = configuration(args, None if updating else package, modules)
        if updating:
            if len(modules) != 1:
                raise ValueError("Update or roll back exactly one module")
            if args.dry_run:
                print(json.dumps({"module": modules[0], "action": args.action,
                                  "bundle": bundle, "state_directory": str(state)},
                                 indent=2))
                return 0
            if args.action == "update":
                if modules != [bundle["module"]]:
                    raise ValueError("Update package belongs to another module")
                module_updates.update(state, args.bundle.resolve(), settings,
                                      args.wait_timeout, run, write_json)
            else:
                if __package__:
                    from . import module_updates
                else:
                    import module_updates
                module_updates.rollback(state, modules[0], settings,
                                        args.wait_timeout, run, write_json)
            return 0
        if args.action == "up" and not args.dry_run:
            if ("contour" in modules
                    and not (state / "models/contour/contour.pt").is_file()):
                raise ValueError(
                    "Run prepare-models for Contour before starting its module"
                )
            if "system1" in modules and not any(
                (state / "models/system1/huggingface").glob(
                    "models--ZhuYuhan--VideoJev-4B/snapshots/*/config.json"
                )
            ):
                raise ValueError(
                    "Run prepare-models for System 1 before starting its module"
                )
        if ("bbox" in modules and args.action in ("build", "prepare-models", "up")
                and not args.dry_run
                and (settings["bbox_device"] == "auto" or args.platform
                     or "bbox_platform" in settings)):
            if __package__:
                from .docker_target import resolve_bbox
            else:
                from docker_target import resolve_bbox
            settings, selection = resolve_bbox(
                settings, run, platform=args.platform,
                building=args.action == "build" or args.build,
            )
            print("Bbox target: " + json.dumps(selection))
        documents = {
            module: spec.compose(module, settings, source, state)
            for module in modules
            if module != "gpu-runtime"
        }
        bases = [
            spec.runtime_build(source, settings, version)
            for version in spec.runtime_versions(modules, settings)
        ]
        if args.action == "plan" or args.dry_run:
            print(
                json.dumps(
                    {
                        "state_directory": str(state),
                        "settings": settings,
                        "base_builds": bases,
                        "compose_projects": documents,
                        "bbox_auto_detection_pending": (
                            "bbox" in modules and settings["bbox_device"] == "auto"
                            and args.action == "plan"
                        ),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        if args.action in ("status", "stop", "logs"):
            # These actions use the last applied files, never silently reconfigure.
            for module in documents:
                if not (state / f"compose.{module}.json").is_file():
                    raise ValueError("Run configure or up for this module first")
                operation = {
                    "status": ["ps", "--all"],
                    "stop": ["stop"],
                    "logs": ["logs", "--tail", "100"],
                }[args.action]
                run(compose_command(state, module, *operation))
            return 0
        initialize(state, modules)
        write_json(state / "settings.json", settings)
        for module, document in documents.items():
            write_json(state / f"compose.{module}.json", document)
        if args.action == "configure":
            print(f"Configuration saved: {state}")
            return 0
        # Compose validation is offline; it precedes every build or startup.
        for module in documents:
            run(compose_command(state, module, "config", "--quiet"))
        if args.action == "build" or (args.action == "up" and args.build):
            for command in bases:
                if "gpu-runtime" not in modules:
                    image = command[command.index("-t") + 1]
                    present = subprocess.run(
                        ["docker", "image", "inspect", image],
                        capture_output=True, text=True,
                    )
                    if present.returncode == 0:
                        print(f"Reusing shared GPU runtime: {image}")
                        continue
                    run(["docker", "info"], stdout=subprocess.DEVNULL)
                run(command)
            for module in documents:
                run(compose_command(state, module, "build"))
            if (args.action == "up" and "bbox" in modules
                    and settings["bbox_device"] == "auto"
                    and spec.bbox_device(settings) == "cuda"):
                # A first source build can only probe the driver beforehand.
                # Check the new PyTorch image's actual GPU kernel support too.
                checked, selection = resolve_bbox(settings, run)
                if spec.bbox_device(checked) == "cpu":
                    settings = checked
                    documents["bbox"] = spec.compose("bbox", settings, source, state)
                    write_json(state / "settings.json", settings)
                    write_json(state / "compose.bbox.json", documents["bbox"])
                    run(compose_command(state, "bbox", "config", "--quiet"))
                    run(compose_command(state, "bbox", "build"))
                    print("Bbox target after GPU kernel probe: "
                          + json.dumps(selection))
        if args.action == "prepare-models":
            for module, document in documents.items():
                if module in ("bbox", "system1", "contour"):
                    run(prepare_command(module, document, state, source))
        if args.action == "up":
            if not documents:
                raise ValueError(
                    "The shared GPU package uses build; it has no running container"
                )
            if (
                "contour" in documents
                and not (state / "models/contour/contour.pt").is_file()
            ):
                raise ValueError(
                    "Run prepare-models for Contour before starting its module"
                )
            if "contour" in documents:
                check_contour_model_access(state / "models/contour/contour.pt")
            if "system1" in documents and not any(
                (state / "models/system1/huggingface").glob(
                    "models--ZhuYuhan--VideoJev-4B/snapshots/*/config.json"
                )
            ):
                raise ValueError(
                    "Run prepare-models for System 1 before starting its module"
                )
            ensure_network(settings)
            # Start workers first when installing a group. Main remains idle.
            for module in sorted(documents, key=lambda item: item == "main"):
                run(
                    compose_command(
                        state,
                        module,
                        "up",
                        "-d",
                        "--no-build",
                        "--pull",
                        "never",
                        "--wait",
                        "--wait-timeout",
                        str(args.wait_timeout),
                    )
                )
            print(
                "Selected modules are healthy. "
                "Open the main console to configure and start analysis."
            )
        return 0
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"Deployment not completed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
