"""Reduce a verified full ZIP to installation files, without changing images."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import zipfile

import yaml


def package(source: Path) -> dict:
    partial = source.with_suffix(".customer.partial")
    with zipfile.ZipFile(source) as old:
        image_records = [n for n in old.namelist() if n.endswith("/IMAGES.json")]
        if len(image_records) != 1:
            raise ValueError("Expected one verified image record")
        prefix = image_records[0].removesuffix("IMAGES.json")
        metadata = json.loads(old.read(image_records[0]))
        compose = yaml.safe_load(old.read(prefix + "compose.yaml"))
        mounts = compose["services"]["harness"]["volumes"]
        config_mount = "./config.delivery.yaml:/app/config.yaml:ro"
        if mounts.count(config_mount) != 1:
            raise ValueError("Unexpected configuration mount")
        mounts.remove(config_mount)
        launcher = old.read(prefix + "start_delivery.py").decode("utf-8").replace("\r\n", "\n")
        config_check = '    find_file(directory, None, ("config.delivery.yaml",))\n'
        if launcher.count(config_check) != 1:
            raise ValueError("Unexpected launcher configuration check")
        tree = ast.parse(launcher.replace(config_check, ""))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.body and isinstance(node.body[0], ast.Expr):
                    value = node.body[0].value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        node.body.pop(0)
                        if not node.body:
                            node.body.append(ast.Pass())
        minimal_launcher = ast.unparse(tree) + "\n"
        compile(minimal_launcher, "start_delivery.py", "exec")
        instructions = """用哪些文件

使用本 ZIP 解压后的全部 5 个文件，放在同一个目录，不要改名。

怎么安装

1. 准备 Docker（Linux 容器模式、Compose v2）和 Python 3.10 或以上。
2. 解压本包，在解压目录执行：
   docker load -i application.tar.gz
   Windows：py -3 start_delivery.py
   Linux：python3 start_delivery.py
3. 打开 http://localhost:8080，填写访问密钥，配置视频源后手动开始分析。

明确使用 CPU 时，在启动命令后加 --mode cpu。
使用 GPU 时需事先配置显卡驱动和 Docker 的 GPU 支持。
模型权重和访问密钥需另行准备；已有安装请保留原数据卷。
"""
        small_files = {
            "compose.yaml": yaml.safe_dump(compose, sort_keys=False),
            "compose.gpu.yaml": yaml.safe_dump(
                yaml.safe_load(old.read(prefix + "compose.gpu.yaml")), sort_keys=False,
            ),
            "start_delivery.py": minimal_launcher,
            "使用说明.txt": instructions,
        }
        with zipfile.ZipFile(partial, "x", allowZip64=True) as new:
            with old.open(prefix + metadata["archive"]) as image:
                with new.open(prefix + "application.tar.gz", "w", force_zip64=True) as target:
                    while block := image.read(4 * 1024 * 1024):
                        target.write(block)
            for name, content in small_files.items():
                new.writestr(prefix + name, content.encode("utf-8"))
    partial.replace(source)
    source.with_suffix(".zip.sha256").unlink(missing_ok=True)
    return {"files": sorted(small_files) + ["application.tar.gz"],
            "bytes": source.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    print(json.dumps(package(parser.parse_args().source), ensure_ascii=False, indent=2))
