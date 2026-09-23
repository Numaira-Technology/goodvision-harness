"""Build a source-only update for verified main application releases."""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
BASE_ID = "sha256:03658e7552eed143aa9bbba1ae42321d811cb9bc5abd0fe3a1962170e6beba1b"
LOCAL_BASE_IDS = (
    "sha256:a805e6a6ea754590f064d74bb5d1c0ffa8521428ba675c08e65cc14b366de604",
    "sha256:8b1547f1ffa3ac3fdaca642cb0e38839184540099180d442532cf51a6b158e84",
    "sha256:62b4cb35e295bc15a7330733a07ee91b82a6fb691494c5eeeb53545e24dbfb4b",
    # Installed cloud-health3 from this release's validated update chain.
    "sha256:5db23e0fee21f6d80ebdac14294232fa68864745ae6c3d3ba56c05387b051228",
)
WINDOWS_BASE_IDS = (
    "sha256:64a7094c7a056d25197fc10bbf4e247c65375adcdafb3d47997a43672c4f5689",
    "sha256:6d8db2f739b196fb7ce8a71484a3ec199b7303b8986a0196a8a1174daf543cbf",
    "sha256:ad277142335b228e93629f921df0419684126fc194b6a9410fcd19d466bb4867",
    # Lemon's diagnostics2 update of the 64a709 Windows delivery. Updates
    # produce a new image ID; the launcher metadata must still be validated.
    "sha256:8bcb5a4c5330106878785d00784043d5d6c7aadbace113b4f0f736b71d8e9f20",
)


# Use the committed application inventory, never a recursive working-directory
# copy: local galleries, model caches, bytecode and recordings live nearby.
def application_files():
    tracked = (
        subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={REPO.as_posix()}",
                "ls-files",
                "-z",
                "src/goodvision_harness",
            ],
            cwd=REPO,
        )
        .decode("utf-8")
        .split("\0")
    )
    prefix = "src/goodvision_harness/"
    return tuple(name[len(prefix) :] for name in tracked if name.startswith(prefix))


def package(output, release, *, record_release=False):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", release):
        raise ValueError("Release must be a valid Docker image tag.")
    files = {
        f"payload/{name}": (REPO / "src/goodvision_harness" / name).read_bytes()
        for name in application_files()
    }
    sources = {
        name[8:]: hashlib.sha256(data).hexdigest() for name, data in files.items()
    }
    history_path = ROOT / "published_sources.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if release in history and history[release] != sources:
        raise ValueError(
            "This release already identifies different application sources; "
            "choose a new release name."
        )
    history = history | {release: sources}
    for name in (
        "apply_update.py",
        "release_metadata.py",
        "verify_installation.py",
        "update_harness.ps1",
        "README.md",
    ):
        files[name] = (ROOT / name).read_bytes()
    files["Dockerfile"] = (
        f"ARG BASE_IMAGE=goodvision-harness-update-base:{release}\n"
        "FROM ${BASE_IMAGE}\n"
        "COPY --chown=10001:10001 . "
        "/usr/local/lib/python3.11/site-packages/goodvision_harness/\n"
        "COPY --chown=10001:10001 . /app/src/goodvision_harness/\n"
        "ENV LOCAL_CAMERA_TARGET_FPS=15 SCENE_ENGINE_MODE=active "
        "SCENE_INDEPENDENT_VLM=true\n"
        f'LABEL goodvision.ui.release="{release}"\n'
    ).encode()
    for name, option in (
        ("UPDATE.cmd", ""),
        ("CHECK.cmd", " -CheckOnly"),
        (
            "UPDATE-8080.cmd",
            ' -Containers "goodvision-live-latest,goodvision-contour-camera-02"',
        ),
        (
            "CHECK-8080.cmd",
            ' -Containers "goodvision-live-latest,goodvision-contour-camera-02"'
            " -CheckOnly",
        ),
    ):
        files[name] = (
            "@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass "
            '-File "%~dp0update_harness.ps1" -PackageDir "%~dp0."' + option + "\r\n"
            "set UPDATE_EXIT=%ERRORLEVEL%\r\npause\r\nexit /b %UPDATE_EXIT%\r\n"
        ).encode()
    manifest = {
        "product": "GoodVision Harness Code Update",
        "release": release,
        "base_id": BASE_ID,
        "accepted_base_ids": [BASE_ID, *WINDOWS_BASE_IDS, *LOCAL_BASE_IDS],
        "published_sources": history,
        "windows_base_ids": list(WINDOWS_BASE_IDS),
        "image": f"goodvision-harness:{release}",
        "includes_models": False,
        "requires_downloads": False,
        "continuous_playback_vlm": True,
        "source_commit": subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO.as_posix()}", "rev-parse", "HEAD"],
            cwd=REPO,
        )
        .decode()
        .strip(),
        "source_modified_files": subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={REPO.as_posix()}",
                "diff",
                "HEAD",
                "--name-only",
                "--",
                "src/goodvision_harness",
                "tools/harness_update",
            ],
            cwd=REPO,
        )
        .decode()
        .splitlines(),
        "files": [
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in sorted(files.items())
        ],
    }
    files["manifest.json"] = json.dumps(manifest, indent=2).encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, data in files.items():
            archive.writestr(f"goodvision-harness-update-{release}/{name}", data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    if record_release:
        # Failed builds must not become trusted installation history.
        temporary = history_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        temporary.replace(history_path)
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": digest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="New immutable release name")
    args = parser.parse_args()
    print(
        json.dumps(
            package(
                REPO / "dist" / f"goodvision-harness-update-{args.release}.zip",
                args.release,
                record_release=True,
            ),
            indent=2,
        )
    )
