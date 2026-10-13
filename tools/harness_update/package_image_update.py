"""Build an offline image update; customer ZIPs contain no application payload."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from package_update import BASE_ID, LOCAL_BASE_IDS, WINDOWS_BASE_IDS, application_files

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
IMAGE = "goodvision-harness:20260923-image1"
RELEASE = "20260923-full1"


def prepare(context):
    context.mkdir(parents=True, exist_ok=True)
    for name in application_files():
        target = context / "application" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/goodvision_harness" / name, target)
    helper = context / "updater"
    helper.mkdir(exist_ok=True)
    for name in ("apply_update.py", "release_metadata.py", "verify_installation.py"):
        shutil.copyfile(ROOT / name, helper / name)
    history = json.loads((ROOT / "published_sources.json").read_text(encoding="utf-8"))
    manifest = {
        "product": "GoodVision Harness Image Update", "release": RELEASE,
        "base_id": BASE_ID,
        "accepted_base_ids": [BASE_ID, *WINDOWS_BASE_IDS, *LOCAL_BASE_IDS],
        "windows_base_ids": list(WINDOWS_BASE_IDS), "published_sources": history,
        "image": IMAGE, "continuous_playback_vlm": True,
        # Existing records are needed to maintain old Windows launcher metadata.
        "files": [{"path": "payload/" + name, "sha256": value}
                  for name, value in history[RELEASE].items()],
    }
    (helper / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (context / "Dockerfile").write_text(f"""FROM goodvision-harness:20260923-full1
USER root
COPY --chown=10001:10001 application/ /app/src/goodvision_harness/
COPY --chown=10001:10001 application/ /usr/local/lib/python3.11/site-packages/goodvision_harness/
COPY updater/ /opt/goodvision/update/
LABEL goodvision.ui.release="{RELEASE}"
ENV LOCAL_CAMERA_TARGET_FPS=15 SCENE_ENGINE_MODE=active SCENE_INDEPENDENT_VLM=true
USER 10001:10001
""", encoding="utf-8")


def package(output):
    partial = output.with_suffix(".image.partial")
    prefix = f"goodvision-harness-update-{RELEASE}/"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        with subprocess.Popen(["docker", "image", "save", IMAGE],
                              stdout=subprocess.PIPE) as process:
            try:
                with archive.open(prefix + "application.tar", "w", force_zip64=True) as target:
                    shutil.copyfileobj(process.stdout, target, 4 * 1024 * 1024)
                if process.wait() != 0:
                    raise RuntimeError("Image export failed")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        archive.writestr(prefix + "UPDATE.cmd", (
            '@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass '
            '-File "%~dp0update.ps1" -PackageDir "%~dp0."\r\n'
            'set UPDATE_EXIT=%ERRORLEVEL%\r\npause\r\nexit /b %UPDATE_EXIT%\r\n'
        ))
        # This is Docker installation orchestration, not application source.
        script = (ROOT / "update_image.ps1").read_text(encoding="utf-8")
        script = "\n".join(line for line in script.splitlines()
                           if not line.lstrip().startswith("#") or line.startswith("#Requires"))
        archive.writestr(prefix + "update.ps1", script.encode("utf-8-sig"))
        archive.writestr(prefix + "使用说明.txt", """用哪些文件

使用本 ZIP，解压后全部文件放在同一个目录，不要改名。

怎么安装

1. 在已安装本程序的 Windows 电脑上，停止全部分析并关闭自动开始分析。
2. 保持 Docker 运行，将 ZIP 解压到新文件夹，不要覆盖原安装目录。
3. 双击 UPDATE.cmd，等待出现 Update ready。
4. 刷新原控制台，连接视频并手动开始分析。原配置、模型和数据保留。

报错时停止操作，联系交付人员。
""".encode("utf-8"))
    partial.replace(output)
    print(json.dumps({"path": str(output), "bytes": output.stat().st_size}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    if args.output:
        package(args.output)
