"""Package an existing update with a minimal customer entrypoint."""

import argparse
import json
from pathlib import Path
import zipfile


def package(source):
    partial = source.with_suffix(".customer.partial")
    with zipfile.ZipFile(source) as old:
        manifest_name = next(n for n in old.namelist() if n.endswith("/manifest.json"))
        prefix = manifest_name.removesuffix("manifest.json")
        manifest = json.loads(old.read(manifest_name))
        manifest.pop("source_commit", None)
        manifest.pop("source_modified_files", None)
        manifest["files"] = [f for f in manifest["files"]
                             if f["path"] != "README.md" and not f["path"].endswith(".cmd")]
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED) as new:
            for f in manifest["files"]:
                new.writestr(prefix + "runtime/" + f["path"], old.read(prefix + f["path"]))
            new.writestr(prefix + "runtime/manifest.json", json.dumps(manifest).encode())
            new.writestr(prefix + "UPDATE.cmd", (
                '@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass '
                '-File "%~dp0runtime\\update_harness.ps1" '
                '-PackageDir "%~dp0runtime"\r\n'
                'set UPDATE_EXIT=%ERRORLEVEL%\r\npause\r\nexit /b %UPDATE_EXIT%\r\n'
            ))
            new.writestr(prefix + "使用说明.txt", """用哪些文件

使用本 ZIP，解压后保留 UPDATE.cmd 和 runtime 文件夹在同一目录。

怎么安装

1. 适用于已安装本程序的 Windows 电脑。先停止全部分析，关闭自动开始分析。
2. 将 ZIP 解压到新文件夹，不要覆盖原安装目录，保持 Docker 运行。
3. 双击 UPDATE.cmd，等待出现 Update ready，然后刷新原控制台。
4. 重新连接视频并手动开始分析。

原配置、模型和数据保留。安装报错时停止操作，联系交付人员。
""".encode("utf-8"))
    partial.replace(source)
    source.with_suffix(".zip.sha256").unlink(missing_ok=True)
    print(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    package(parser.parse_args().source)
