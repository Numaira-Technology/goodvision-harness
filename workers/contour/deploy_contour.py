#!/usr/bin/env python3
"""Deploy the standalone Linux amd64 release using Python's standard library."""
# Chinese operator-facing messages intentionally use Chinese punctuation.
# ruff: noqa: RUF001

import argparse
import hashlib
import ipaddress
import json
import os
import platform
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

IMAGES = ("goodvision-contour:0.1.0", "goodvision-contour-runtime:0.1.0")
MODEL_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
MODEL_BYTES = 3502755717
READY_PROBE = """
import json, pathlib, urllib.request
key = pathlib.Path('/run/secrets/node_key').read_text().strip()
request = urllib.request.Request('http://127.0.0.1:8093/health/ready',
    headers={'Authorization': 'Bearer ' + key})
result = json.load(urllib.request.urlopen(request, timeout=8))
assert result.get('ready') is True and result.get('gpu_ready') is True
"""
GPU_PROBE = """
from pathlib import Path
import torch
with Path('/models/contour.pt').open('rb') as checkpoint:
    assert checkpoint.read(1), 'Mounted checkpoint is empty'
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.ones(64, device='cuda').sum().item() == 64
p = torch.cuda.get_device_properties(0)
print('GPU: %s; VRAM: %.1f GiB; CUDA: %s' %
      (p.name, p.total_memory / 1024**3, torch.version.cuda))
"""


def say(message):
    print(message, flush=True)


def run(command, *, capture=False, check=True, **kwargs):
    return subprocess.run(
        [str(part) for part in command],
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        **kwargs,
    )


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_sidecar(path):
    sidecar = Path(str(path) + ".sha256")
    if sidecar.is_file():
        say("校验文件：" + path.name)
        expected = sidecar.read_text(encoding="utf-8-sig").split()[0].lower()
        if digest(path) != expected:
            raise RuntimeError("SHA-256 校验失败：" + str(path))


def release_root(source):
    if (source / "workers/contour/compose.yaml").is_file():
        return source
    if (source / "compose.yaml").is_file() and source.name == "contour":
        return source.parent.parent
    return None


def unpack(source, destination):
    """Extract into a new directory; never overwrite an existing installation."""
    archive = next(
        (
            source / name
            for name in (
                "goodvision-contour-with-model.zip",
                "goodvision-contour-framework.zip",
            )
            if (source / name).is_file()
        ),
        None,
    )
    if archive is None:
        raise RuntimeError("找不到模型 ZIP 或已解压的 workers/contour 目录。")
    verify_sidecar(archive)
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("release-manifest.json"))
        if manifest.get("product") != "GoodVision Contour":
            raise RuntimeError("这不是 GoodVision Contour 部署包。")
        entries = {}
        for item in manifest["files"]:
            name = item["path"]
            parts = PurePosixPath(name).parts
            if (
                not parts
                or parts[0] != "workers"
                or ".." in parts
                or "\\" in name
                or ":" in name
                or name in entries
            ):
                raise RuntimeError("部署包存在不安全或重复的路径。")
            info = bundle.getinfo(name)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise RuntimeError("部署包不能包含符号链接。")
            if info.file_size != item["bytes"]:
                raise RuntimeError("部署包文件长度不符：" + name)
            entries[name] = item
        if destination.exists():
            record = destination / "release-manifest.json"
            if not release_root(destination) or not record.is_file():
                raise RuntimeError(
                    "目标目录已存在且不是完整部署；请指定新的 --install-dir。"
                )
            installed = json.loads(record.read_text(encoding="utf-8"))
            if installed != manifest:
                raise RuntimeError(
                    "目标目录属于另一份部署包；请指定新的 --install-dir。"
                )
            for name, item in entries.items():
                target = destination / name
                if (
                    not target.is_file()
                    or target.is_symlink()
                    or digest(target) != item["sha256"]
                ):
                    raise RuntimeError("已有部署文件被修改或缺失：" + name)
            say("沿用已有安装目录：" + str(destination))
            return destination
        required = sum(item["bytes"] for item in entries.values())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination.parent).free < required + 512 * 1024**2:
            raise RuntimeError("解压目录磁盘空间不足。")
        destination.mkdir(mode=0o700)
        say("解压部署包：" + str(destination))
        for name, item in entries.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            parent = target.parent
            while parent != destination:
                parent.chmod(0o755)
                parent = parent.parent
            with bundle.open(name) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
            target.chmod(0o644)
            if digest(target) != item["sha256"]:
                raise RuntimeError("解压校验失败：" + name)
        (destination / "release-manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    return destination


def verify_model(models):
    for name in ("contour.pt", "model-manifest.json", "MODEL_LICENSE.txt"):
        if not (models / name).is_file():
            raise RuntimeError("缺少模型文件：" + str(models / name))
    origin = json.loads((models / "model-manifest.json").read_text(encoding="utf-8"))
    model = models / "contour.pt"
    if (
        origin.get("sha256") != MODEL_SHA256
        or origin.get("bytes") != MODEL_BYTES
        or model.stat().st_size != MODEL_BYTES
        or digest(model) != MODEL_SHA256
    ):
        raise RuntimeError("模型完整性校验失败；不会启动推理。")


def initialize_key(node):
    directory = node / "secrets"
    directory.mkdir(mode=0o700, exist_ok=True)
    key = directory / "node-key"
    if key.is_symlink():
        raise RuntimeError("访问密钥不能是符号链接。")
    try:
        fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if not key.read_text().strip():
            raise RuntimeError("已有访问密钥为空；请检查 " + str(key)) from None
    else:
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(36) + "\n")
        # Compose mounts this file into a container running as UID 10004.
        key.chmod(0o644)
    return key


def write_env(node, values):
    def quoted(value):
        if "\n" in value or "\r" in value:
            raise RuntimeError("配置不能包含换行符。")
        return "'" + value.replace("'", "\\'") + "'"

    content = "".join(
        name + "=" + quoted(value) + "\n" for name, value in values.items()
    )
    path = node / ".env"
    if path.is_symlink():
        raise RuntimeError(".env 不能是符号链接。")
    if path.exists() and path.read_text() != content:
        backup = node / (".env.before-deploy-" + str(time.time_ns()))
        shutil.copy2(path, backup)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def await_ready(compose, env, timeout):
    deadline = time.monotonic() + timeout
    announced = 0.0
    while time.monotonic() < deadline:
        probe = run(
            [*compose, "exec", "-T", "contour", "python", "-c", READY_PROBE],
            env=env,
            capture=True,
            check=False,
            timeout=20,
        )
        if probe.returncode == 0:
            return
        container = run(
            [*compose, "ps", "-a", "-q", "engine"], env=env, capture=True, timeout=20
        ).stdout.strip()
        if container:
            state = json.loads(
                run(
                    ["docker", "inspect", "--format", "{{json .State}}", container],
                    env=env,
                    capture=True,
                    timeout=20,
                ).stdout
            )
            if state.get("OOMKilled") or state.get("Status") in (
                "exited",
                "dead",
                "restarting",
            ):
                raise RuntimeError("推理容器已退出或重启，请查看日志和 GPU 剩余显存。")
        if time.monotonic() >= announced:
            say("控制台已启动，正在等待模型加载和 GPU 预热……")
            announced = time.monotonic() + 30
        time.sleep(5)
    raise RuntimeError("等待模型就绪超时；控制台存活不代表模型可用。")


def deploy(args):
    if platform.system() != "Linux" or platform.machine().lower() not in (
        "x86_64",
        "amd64",
    ):
        raise RuntimeError(
            "本脚本用于 Linux Intel/AMD + NVIDIA GPU 服务器，不适用于 Jetson/ARM。"
        )
    if not shutil.which("docker"):
        raise RuntimeError(
            "请先安装 Docker Engine、Docker Compose 和 NVIDIA Container Toolkit。"
        )
    run(["docker", "compose", "version"], timeout=30)
    info = json.loads(
        run(
            ["docker", "info", "--format", "{{json .}}"], capture=True, timeout=30
        ).stdout
    )
    if info.get("OSType") != "linux" or info.get("Architecture") not in (
        "x86_64",
        "amd64",
    ):
        raise RuntimeError("Docker 服务端必须为 Linux amd64。")
    # Bind mounts and GPU selection must target this server, not another Docker host.
    context = json.loads(
        run(["docker", "context", "inspect"], capture=True, timeout=30).stdout
    )
    endpoint = os.environ.get("DOCKER_HOST", context[0]["Endpoints"]["docker"]["Host"])
    if not endpoint.startswith("unix://"):
        raise RuntimeError("请使用本机 Docker 服务，不能通过远程 Docker context 部署。")
    with socket.socket() as sock:
        sock.bind((args.bind, 0))
    source = args.package_dir.expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError("部署包目录不存在：" + str(source))
    destination = (
        (args.install_dir or source / "goodvision-contour").expanduser().resolve()
    )
    root = release_root(source) or unpack(source, destination)
    node = root / "workers/contour"
    models = (args.models or node / "models").expanduser().resolve()
    say("校验模型完整性……")
    verify_model(models)
    values = {
        "CONTOUR_BIND_ADDRESS": args.bind,
        "CONTOUR_PORT": str(args.port),
        "CONTOUR_GPU_ID": args.gpu,
        "CONTOUR_CAPACITY": str(args.capacity),
        "CONTOUR_MODELS_DIR": str(models),
        "CONTOUR_TORCH_CUDA": args.cuda,
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CONTOUR_")
    }
    env.update(values)
    key = initialize_key(node)
    write_env(node, values)
    compose = [
        "docker",
        "compose",
        "--project-name",
        "goodvision-contour",
        "--project-directory",
        str(node),
        "--env-file",
        str(node / ".env"),
        "-f",
        str(node / "compose.yaml"),
        "-f",
        str(node / "compose.gpu.yaml"),
    ]
    run([*compose, "config", "--quiet"], env=env, timeout=30)
    image_archive = source / "goodvision-contour-images.tar.gz"
    if args.build:
        say("按 --build 参数联网构建运行镜像；不会下载模型。")
        run([*compose, "build"], env=env)
    elif image_archive.is_file():
        verify_sidecar(image_archive)
        say("导入离线镜像……")
        run(["docker", "load", "-i", image_archive], env=env)
    for image in IMAGES:
        inspect = run(
            ["docker", "image", "inspect", image], env=env, capture=True, check=False
        )
        if inspect.returncode:
            raise RuntimeError(
                "缺少 goodvision-contour-images.tar.gz；"
                "请复制该文件，或加 --build 联网构建。"
            )
        metadata = json.loads(inspect.stdout)[0]
        if metadata.get("Architecture") != "amd64" or metadata.get("Os") != "linux":
            raise RuntimeError("镜像架构不兼容：" + image)
    say("检查容器内 CUDA 是否可用……")
    run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--gpus",
            "device=" + args.gpu,
            "--volume",
            str(models) + ":/models:ro",
            "--entrypoint",
            "python",
            IMAGES[1],
            "-c",
            GPU_PROBE,
        ],
        env=env,
        timeout=120,
    )
    try:
        run([*compose, "up", "-d", "--no-build", "--pull", "never"], env=env)
        await_ready(compose, env, args.timeout)
    except (Exception, KeyboardInterrupt):
        say("部署尚未通过就绪检查，停止本次推理服务，保留控制台和文件供排查。")
        run([*compose, "stop", "-t", "30", "engine"], env=env, check=False, timeout=60)
        run(
            [*compose, "logs", "--tail", "60", "engine"],
            env=env,
            check=False,
            timeout=30,
        )
        raise
    say("\n部署成功：模型加载和 GPU 预热已通过（持续视频业务仍需验收）。")
    host = args.bind if args.bind != "0.0.0.0" else "<服务器IP>"
    say(f"控制台：http://{host}:{args.port}/")
    say("访问密钥文件：" + str(key))
    say("管理目录：" + str(node))


def main():
    parser = argparse.ArgumentParser(
        description="GoodVision Contour 一键部署（Linux amd64）"
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path.cwd(),
        help="ZIP/镜像所在目录或已解压目录",
    )
    parser.add_argument(
        "--install-dir", type=Path, help="ZIP 解压目标，默认包目录/goodvision-contour"
    )
    parser.add_argument(
        "--models", type=Path, help="已有模型目录；含模型 ZIP 不需要此参数"
    )
    parser.add_argument("--bind", required=True, help="服务器实际持有的 IPv4 地址")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--gpu", default="0", help="GPU 序号或 GPU UUID，默认 0")
    parser.add_argument(
        "--capacity", type=int, default=1, help="独立视频通道数，默认 1"
    )
    parser.add_argument(
        "--build", action="store_true", help="联网构建镜像，代替离线导入"
    )
    parser.add_argument(
        "--cuda", choices=("cu126", "cu128"), default="cu128", help="只对 --build 生效"
    )
    parser.add_argument("--timeout", type=int, default=900, help="等待模型就绪的秒数")
    args = parser.parse_args()
    ipaddress.IPv4Address(args.bind)
    if (not 1 <= args.port <= 65535 or args.timeout < 1
            or "," in args.gpu or args.capacity < 1):
        parser.error("端口、等待时间或单 GPU 参数无效。")
    try:
        deploy(args)
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as exc:
        say("\n部署失败：" + str(exc))
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            say(exc.stderr.strip())
        return 1
    except KeyboardInterrupt:
        say("\n已中断部署。")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
