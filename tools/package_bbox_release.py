"""One offline Bbox update ZIP; the installer selects architecture and device."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

if __package__:
    from . import module_updates as updates
    from .package_image_update import package as image_package
else:
    import module_updates as updates
    from package_image_update import package as image_package

TARGETS = {"cpu-arm64": "linux/arm64/cpu", "cpu-amd64": "linux/amd64/cpu",
           "cuda-amd64": "linux/amd64/cuda"}


def package(version, variants, output, *, assets=()):
    output.mkdir(parents=True, exist_ok=False)
    (output / "variants").mkdir()
    catalog = {"schema": 2, "module": "bbox", "version": version,
               "variants": {}, "gpu_probe_images": []}
    with tempfile.TemporaryDirectory(prefix="goodvision-bbox-release-") as work:
        for target, (image, baseline) in variants.items():
            folder = Path(work) / target
            manifest = image_package("bbox", version, [image], folder,
                                     baseline, assets)
            record = manifest["images"]["bbox"]
            platform = record["os"] + "/" + record["architecture"]
            if not TARGETS[target].startswith(platform + "/"):
                raise ValueError("Bbox image does not match its variant platform")
            path = output / "variants" / (target + ".zip")
            shutil.move(str(folder.with_suffix(".zip")), path)
            catalog["variants"][TARGETS[target]] = {
                "file": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size, "sha256": updates.digest(path),
            }
            if target == "cuda-amd64":
                catalog["gpu_probe_images"] = [
                    record["name"], *[entry["name"] for entry in manifest["requires"]],
                ]
            # The launchers are identical; no application image carries them.
            for filename in ("install.py", "docker_modules.py", "module_updates.py",
                             "docker_target.py", "rename.cmd", "update.cmd",
                             "rollback.cmd"):
                shutil.copyfile(folder / filename, output / filename)
    (output / "update.json").write_text(json.dumps(catalog, indent=2) + "\n")
    helpers = {name: updates.digest(output / name) for name in (
        "install.py", "docker_modules.py", "module_updates.py", "docker_target.py",
    )}
    (output / "module-package.json").write_text(json.dumps({
        "module": "bbox", "version": version, "bbox_device": "auto",
        "delivery": "image-update", "files": helpers,
    }, indent=2) + "\n")
    (output / "UPDATE.txt").write_text(
        "Bbox update: python install.py update --bundle update.json\n"
        "Windows: update.cmd\nRollback: python install.py rollback\n"
        "The Docker engine architecture and container GPU access select the variant.\n"
        "Existing explicit CPU/CUDA policy is preserved. Use --bbox-device auto "
        "to enable automatic selection on an older installation.\n"
        "Keep the original --namespace and --state-dir for this deployment.\n",
    )
    updates.read_bundle(output / "update.json")
    zipped = output.with_suffix(".zip")
    partial = zipped.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output).as_posix())
    partial.replace(zipped)
    print(json.dumps({"package": str(zipped), "bytes": zipped.stat().st_size,
                      "variants": list(catalog["variants"])}), flush=True)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset", action="append", default=[])
    for target in TARGETS:
        parser.add_argument("--" + target, required=True)
        parser.add_argument("--" + target + "-baseline", type=Path)
    args = parser.parse_args()
    variants = {target: (getattr(args, target.replace("-", "_")),
                         getattr(args, target.replace("-", "_") + "_baseline"))
                for target in TARGETS}
    assets = [(value.split("=", 1)[0], Path(value.split("=", 1)[1]))
              for value in args.asset]
    package(args.version, variants, args.output, assets=assets)


if __name__ == "__main__":
    main()
