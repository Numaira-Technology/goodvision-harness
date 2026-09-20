"""Build a source-only update for verified main application releases."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
BASE_ID = "sha256:03658e7552eed143aa9bbba1ae42321d811cb9bc5abd0fe3a1962170e6beba1b"
RELEASE = "20260921-realtime1"
LOCAL_BASE_IDS = (
    "sha256:a805e6a6ea754590f064d74bb5d1c0ffa8521428ba675c08e65cc14b366de604",
)
WINDOWS_BASE_IDS = (
    "sha256:64a7094c7a056d25197fc10bbf4e247c65375adcdafb3d47997a43672c4f5689",
    "sha256:6d8db2f739b196fb7ce8a71484a3ec199b7303b8986a0196a8a1174daf543cbf",
    "sha256:ad277142335b228e93629f921df0419684126fc194b6a9410fcd19d466bb4867",
)
# Use the committed application inventory, never a recursive working-directory
# copy: local galleries, model caches, bytecode and recordings live nearby.
def application_files():
    tracked = subprocess.check_output(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "ls-files", "-z", "src/goodvision_harness"],
        cwd=REPO,
    ).decode("utf-8").split("\0")
    prefix = "src/goodvision_harness/"
    return tuple(name[len(prefix):] for name in tracked if name.startswith(prefix))


def package(output, release=RELEASE):
    files = {
        f"payload/{name}": (REPO / "src/goodvision_harness" / name).read_bytes()
        for name in application_files()
    }
    for name in (
        "apply_update.py",
        "release_metadata.py",
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
        f'LABEL goodvision.ui.release="{release}"\n'
    ).encode()
    for name, option in (
        ("UPDATE.cmd", ""),
        ("CHECK.cmd", " -CheckOnly"),
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
        "windows_base_ids": list(WINDOWS_BASE_IDS),
        "image": f"goodvision-harness:{release}",
        "includes_models": False,
        "requires_downloads": False,
        "continuous_playback_vlm": True,
        "source_commit": subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO.as_posix()}", "rev-parse", "HEAD"], cwd=REPO,
        ).decode().strip(),
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
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": digest}


if __name__ == "__main__":
    print(
        json.dumps(
            package(REPO / "dist" / f"goodvision-harness-update-{RELEASE}.zip"),
            indent=2,
        )
    )
