"""Export a standalone source installer; never builds images or downloads models.

Run explicitly on the release machine. Only reviewed, tracked module source is
included. The recipient needs Docker Compose and Python 3.10+, but no Git checkout.
"""

# Chinese customer instructions intentionally use Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

if __package__:
    from . import docker_modules as spec
else:
    import docker_modules as spec

ROOT = Path(__file__).resolve().parents[1]
COMMON = (".dockerignore", "pyproject.toml", "README.md")
INSTALLER_FILES = (
    "deploy_modules.py", "docker_modules.py", "module_updates.py", "docker_target.py",
)
SOURCES = {
    "main": (
        *COMMON,
        "Dockerfile",
        "tools/docker_dependencies.py",
        "config.yaml",
        "src/goodvision_harness",
        "workers/__init__.py",
        "workers/detector_rich_label",
    ),
    "bbox": (
        ".dockerignore",
        "workers/__init__.py",
        "workers/bbox",
        "workers/detector_rich_label",
        "workers/gpu_runtime",
    ),
    "system1": (
        *COMMON,
        "src/goodvision_harness",
        "workers/system1",
        "workers/gpu_runtime",
        "tools/videojev_probe/requirements.txt",
        "tools/docker_dependencies.py",
    ),
    "contour": (
        "workers/contour",
        "workers/sam_scene",
        "workers/gpu_runtime",
        ".dockerignore",
    ),
    "gpu-runtime": ("workers/gpu_runtime", ".dockerignore"),
}
REQUIRED = {
    "main": (
        "Dockerfile", "src/goodvision_harness/service.py", "config.yaml",
        "tools/docker_dependencies.py",
    ),
    "bbox": (
        "workers/bbox/Dockerfile",
        "workers/bbox/Dockerfile.cpu",
        "workers/bbox/requirements.txt",
        "workers/detector_rich_label/app.py",
        "workers/gpu_runtime/Dockerfile",
    ),
    "system1": (
        "workers/system1/Dockerfile",
        "tools/docker_dependencies.py",
        "src/goodvision_harness/videojev_service.py",
        "tools/videojev_probe/requirements.txt",
        "workers/gpu_runtime/Dockerfile",
    ),
    "contour": (
        "workers/contour/Dockerfile",
        "workers/contour/Dockerfile.engine",
        "workers/contour/compose.shared-runtime.yaml",
        "workers/sam_scene/requirements.txt",
        "workers/gpu_runtime/Dockerfile",
    ),
    "gpu-runtime": ("workers/gpu_runtime/Dockerfile",),
}


def tracked_files(root, module):
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve().as_posix()}",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--",
            *SOURCES[module],
        ],
        check=True,
        capture_output=True,
    )
    return sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})


def safe_source(root, relative):
    path = root / relative
    resolved = path.resolve()
    if (
        not resolved.is_relative_to(root.resolve())
        or not path.is_file()
        or any(
            part.is_symlink() for part in [path, *path.parents] if part != root.parent
        )
    ):
        raise ValueError(f"Not a regular package source: {relative}")
    parts = Path(relative).parts
    if any(
        part in {"secrets", "models", ".models", ".goodvision", "__pycache__"}
        or part.startswith(".venv")
        for part in parts
    ):
        raise ValueError(f"Local state cannot be packaged: {relative}")
    if path.name.startswith(".env") and not path.name.endswith(".example"):
        raise ValueError(f"Local environment cannot be packaged: {relative}")
    if path.suffix.lower() in {
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".mp4",
        ".pem",
        ".key",
    }:
        raise ValueError(f"Private media/models/keys cannot be packaged: {relative}")
    return path


def instructions(module, version, device):
    build = "python install.py build"
    steps = (
        f"{build}\n"
        if module == "gpu-runtime"
        else f"{build}\n"
        + (
            "python install.py prepare-models\n"
            if module in ("bbox", "system1", "contour")
            else ""
        )
        + "python install.py up\n"
    )
    if module == "bbox" and device == "auto":
        steps = "python install.py up --build\n"
    return f"""# GoodVision {module} · {version}

这是可独立使用的源码安装包，包含本模块构建所需源码，无需克隆 Git 仓库。
本包不含预构建镜像、模型权重、密钥或用户数据。首次构建需要网络。
需要 Python 3.10+、Docker Engine / Docker Desktop Linux 容器及 Docker Compose v2。
GPU 模块另需 NVIDIA 驱动及容器 GPU 支持；System 1 使用 Linux CUDA。
Bbox 安装类型：{device}。共享 GPU 基础镜像按版本复用，不单独启动容器。

先查看计划（不启动 Docker、不修改配置）：

    python install.py plan

正式安装时，逐条执行：

{"".join("    " + line + chr(10) for line in steps.splitlines())}
prepare-models 明确下载模型，可能需要获准的模型访问权限；HF_TOKEN 只从本机
环境读取，不写入包。也可事先将权重复制到 plan 显示的数据目录下的 models/模块名。
System 1 预下载不代表完成 GPU 推理验收。

同机的各个安装包使用相同 --namespace（默认 default），自动共用一张容器网络。
每个模块单独管理容器、日志和数据卷。数据目录默认为用户目录下的
.goodvision/docker/default，与安装包目录分开；升级时保留它及 Docker 数据卷。
自定义 --state-dir 时，同机各包必须使用同一个目录。

主程序默认为 http://127.0.0.1:8080。它保存设置和视频工作流，不会自动开始分析。
同机模块地址自动配置；远端模块可在主程序包执行 configure --bbox-url 或
--system1-url，随后 up 生效，也可在工作流中选择独立服务器/云服务器并保存地址。
Contour 的访问密钥位于数据目录的 secrets/contour-key，在主程序部署设置中保存一次。
既有控制台保存的部署设置/远端地址优先于安装默认值。

    python install.py configure --main-port 18080
    python install.py status
    python install.py logs
    python install.py stop

stop 保留模型、设置、数据卷和共享网络。它仅停止此包的模块。
再次 up 恢复。GPU 共享依赖包只有 build，无需 up/stop。
收到本模块镜像更新 ZIP 后，一条命令完成更新：

    python install.py update --bundle goodvision-{module}-更新版本.zip
    python install.py rollback

更新仅载入本模块的变更层、检查健康状态，保留现有模型、密钥、端口与数据。
健康检查失败自动恢复旧模块；版本和 GPU 底座按模块独立保存。
默认只向宿主机本地发布端口；跨机器连接需显式 --bind 私网IP，并使用受控内网/VPN。
不同机器各有自己的 Docker 网络；远端主程序必须填写服务器地址。

独立源码安装包的生成不代表镜像已构建或真实模型已验收。
"""


def inventory(root, module):
    files = tracked_files(root, module)
    if not set(REQUIRED[module]).issubset(files):
        raise ValueError("Required module sources are missing from the Git index")
    selected = {"source/" + name: safe_source(root, name) for name in files}
    for filename in INSTALLER_FILES:
        target = "install.py" if filename == "deploy_modules.py" else filename
        selected[target] = safe_source(root, "tools/" + filename)
    return selected


def package(root, module, version, device, output):
    spec.name(version)
    if module not in SOURCES or device not in ("auto", "cpu", "cuda"):
        raise ValueError("Unknown module or Bbox device")
    content = {
        name: path.read_bytes() for name, path in inventory(root, module).items()
    }
    content["START_HERE.md"] = instructions(module, version, device).encode("utf-8")
    content["Install.ps1"] = (
        b"$ErrorActionPreference = 'Stop'\n"
        b"if (Get-Command py -ErrorAction SilentlyContinue) {\n"
        b"  & py -3 (Join-Path $PSScriptRoot 'install.py') @args\n"
        b"} elseif (Get-Command python -ErrorAction SilentlyContinue) {\n"
        b"  & python (Join-Path $PSScriptRoot 'install.py') @args\n"
        b"} else { throw 'Install Python 3.10 or newer first.' }\n"
        b"exit $LASTEXITCODE\n"
    )
    manifest = {
        "format": 1,
        "module": module,
        "version": version,
        "bbox_device": device,
        "delivery": "source-installer",
        "includes_images": False,
        "includes_models": False,
        "files": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(content.items())
        },
    }
    content["module-package.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    output = output.resolve()
    if output.exists():
        raise ValueError("Output already exists; use a new release filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation, no cleanup or overwriting of another release's files.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(content.items()):
            archive.writestr(name, data)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=spec.MODULES)
    parser.add_argument("--version", required=True)
    parser.add_argument("--bbox-device", choices=("auto", "cpu", "cuda"),
                        default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Check inputs only; write no package"
    )
    args = parser.parse_args(argv)
    spec.name(args.version)
    if args.check:
        print(
            json.dumps(
                {
                    "module": args.module,
                    "delivery": "source-installer",
                    "files": sorted(inventory(ROOT, args.module)),
                },
                indent=2,
            )
        )
        return
    if args.output is None:
        parser.error("--output is required unless --check is used")
    package(ROOT, args.module, args.version, args.bbox_device, args.output)
    print(f"Source installer written: {args.output}")


if __name__ == "__main__":
    main()
