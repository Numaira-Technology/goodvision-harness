"""Install, supervise and probe one pinned local GPU model; no GPU rental/API calls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from tools.vlm_benchmark.report import write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path("/workspace/goodvision-vlm-qwen38")
PROGRAM = "qwen-vlm"


def profile():
    return json.loads((HERE / "local_model_profile.json").read_text())


def run(command, *, timeout=120, capture=False, stdout_only=False, env=None):
    return subprocess.run(
        list(map(str, command)),
        check=True,
        timeout=timeout,
        text=True,
        capture_output=capture,
        stdout=subprocess.PIPE if stdout_only else None,
        cwd=ROOT,
        env=env,
    )


def parse_gpu(text, index, policy):
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    selected = [r for r in rows if len(r) == 6 and r[0].strip() == str(index)]
    if len(selected) != 1:
        raise RuntimeError("GPU index not found in nvidia-smi")
    idx, name, uuid, total, free, driver = [x.strip() for x in selected[0]]
    if "RTX 4090" not in name:
        raise RuntimeError("this profile requires an RTX 4090")
    if int(driver.split(".")[0]) < policy["min_driver_major"]:
        raise RuntimeError("CUDA 13 engine requires NVIDIA driver R580 or newer")
    if (
        float(total) < policy["min_vram_mib"]
        or float(free) < policy["min_free_vram_mib"]
    ):
        raise RuntimeError("insufficient/free GPU memory; stop other GPU workloads")
    return dict(
        index=int(idx),
        name=name,
        uuid=uuid,
        total_mib=float(total),
        free_mib=float(free),
        driver=driver,
    )


def available_memory(proc=Path("/proc/meminfo"), cgroup=Path("/sys/fs/cgroup")):
    info = dict(line.split(":", 1) for line in proc.read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    for limit_name, usage_name in (
        ("memory.max", "memory.current"),
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes"),
    ):
        limit, usage = cgroup / limit_name, cgroup / usage_name
        if limit.is_file() and usage.is_file():
            text = limit.read_text().strip()
            if text != "max":
                available = min(available, int(text) - int(usage.read_text().strip()))
    return available


def preflight(root, gpu_index, policy):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("requires Linux x86_64, Ubuntu 24.04 or equivalent")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("install requires Python 3.12 with venv/pip support")
    libc, version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 39):
        raise RuntimeError("pinned engine wheels require glibc >=2.39 (Ubuntu 24.04)")
    if available_memory() < policy["min_ram_gib"] * 1024**3:
        raise RuntimeError("requires 32 GiB available host RAM")
    if shutil.disk_usage(root).free < policy["min_free_disk_gib"] * 1024**3:
        raise RuntimeError("requires 60 GiB free disk before installation")
    output = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
        timeout=10,
    ).stdout
    return parse_gpu(output, gpu_index, policy)


@contextmanager
def operation_lock(root):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".operation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "another model installation/start/check is running"
            ) from None
        yield


def clean_environment():
    # Local loopback service: do not inherit remote API keys or vLLM overrides.
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    for name in (
        "INFERENCE_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "LOCAL_INFERENCE_API_KEY",
        "PYTHONPATH",
        "PYTHONHOME",
    ):
        env.pop(name, None)
    return env


def load_install(root):
    data = json.loads((root / "deployment.json").read_text())
    if data["state"] != "installed" or data["profile"] != profile():
        raise RuntimeError(
            "deployment missing or profile changed; use a new model root"
        )
    if "engine_requirements" not in data:
        raise RuntimeError(
            "legacy deployment lacks explicit engine requirements; "
            "review the original installation before migrating its record"
        )
    if data["engine_requirements"] != (HERE / "model-requirements.txt").read_text():
        raise RuntimeError("engine lock changed; use a new model root")
    if data["repo"] != str(ROOT):
        raise RuntimeError("deployment belongs to another checkout")
    return data


def serve_command(root, data):
    p = data["profile"]
    return [
        str(root / "engine/bin/python"),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        data["model_path"],
        "--served-model-name",
        p["served_model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(data["port"]),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--quantization",
        "compressed-tensors",
        "--max-model-len",
        str(p["max_model_len"]),
        "--max-num-seqs",
        str(p["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(p["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        str(p["gpu_memory_utilization"]),
        "--limit-mm-per-prompt",
        '{"image":4,"video":0}',
        "--mm-processor-kwargs",
        '{"max_pixels":518400}',
        "--mm-processor-cache-gb",
        "0",
        "--no-enable-prefix-caching",
        "--cpu-offload-gb",
        "0",
        "--enforce-eager",
        "--generation-config",
        "vllm",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ]


def supervisor_config(root):
    # Supervisor expands percent signs; shlex handles spaces in argument paths.
    e = lambda value: str(value).replace("%", "%%")  # noqa: E731
    worker = shlex.join(
        [
            sys.executable,
            "-m",
            "tools.vlm_benchmark.local_model",
            "_serve",
            "--root",
            str(root),
        ]
    )
    return f"""[unix_http_server]
file={e(root / "supervisor.sock")}
chmod=0600
[supervisord]
logfile={e(root / "supervisor.log")}
pidfile={e(root / "supervisor.pid")}
childlogdir={e(root)}
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://{e(root / "supervisor.sock")}
[program:{PROGRAM}]
directory={e(ROOT)}
command={e(worker)}
autostart=false
autorestart=false
startretries=0
startsecs=1
stopasgroup=true
killasgroup=true
stopwaitsecs=30
redirect_stderr=true
stdout_logfile={e(root / "model.log")}
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=2
"""


def validate_checkpoint(path):
    config = json.loads((path / "config.json").read_text())
    if (
        config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
        or not config.get("vision_config")
        or config.get("language_model_only")
    ):
        raise RuntimeError("checkpoint has no expected multimodal architecture")
    q = config.get("quantization_config", {})
    groups = q.get("config_groups", {})
    if (
        q.get("quant_method") != "compressed-tensors"
        or not groups
        or any(
            group.get("weights", {}).get("num_bits") != 4 for group in groups.values()
        )
    ):
        raise RuntimeError("checkpoint is not the pinned compressed W4A16 model")
    index = path / "model.safetensors.index.json"
    shards = (
        set(json.loads(index.read_text())["weight_map"].values())
        if index.exists()
        else {"model.safetensors"}
    )
    if not shards or any(
        Path(name).name != name or not (path / name).is_file() for name in shards
    ):
        raise RuntimeError("missing/invalid model shards")
    for name in ("tokenizer_config.json", "preprocessor_config.json"):
        if not (path / name).is_file():
            raise RuntimeError(f"missing processor artifact: {name}")


def install(args):
    root, p = args.root, profile()
    marker = root / "deployment.json"
    if marker.exists():
        installed = load_install(root)
        if installed["gpu_index"] != args.gpu or installed["port"] != args.port:
            raise RuntimeError("existing deployment GPU/port differs; use a new root")
        print(f"Already installed: {root}")
        return
    write_json(root / "install-state.json", {"state": "checking_environment"})
    env = clean_environment()
    try:
        gpu = preflight(root, args.gpu, p)
        write_json(root / "install-state.json", {"state": "installing", "gpu": gpu})
        for name, requirements in (
            ("engine", "model-requirements.txt"),
            ("bench", "requirements.txt"),
        ):
            print(f"Installing isolated {name} environment...", flush=True)
            venv = root / name
            run([sys.executable, "-m", "venv", venv])
            python = venv / "bin/python"
            run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--only-binary=:all:",
                    "--no-cache-dir",
                    "-r",
                    HERE / requirements,
                ],
                timeout=2400,
                env=env,
            )
            if name == "bench":
                run([python, "-m", "pip", "install", "--no-deps", "-e", ROOT], env=env)
            run([python, "-m", "pip", "check"], env=env)
            frozen = run([python, "-m", "pip", "freeze"], capture=True, env=env).stdout
            (root / f"{name}-installed.txt").write_text(frozen)
        # Verify actual CUDA execution, not just nvidia-smi or import torch.
        runtime_env = {**env, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
        run(
            [
                root / "engine/bin/python",
                "-c",
                "import torch; assert torch.cuda.is_available(); "
                "x=torch.ones(1,device='cuda'); assert x.item()==1; "
                "print(torch.cuda.get_device_name(0))",
            ],
            env=runtime_env,
        )
        print("Downloading pinned W4A16 model snapshot...", flush=True)
        downloaded = run(
            [
                root / "engine/bin/python",
                "-c",
                "from huggingface_hub import snapshot_download; import sys; "
                "print(snapshot_download(sys.argv[1],revision=sys.argv[2],"
                "cache_dir=sys.argv[3],allow_patterns=['*.json','*.jinja','*.txt',"
                "'*.safetensors','*.model']))",
                p["model_repo"],
                p["model_revision"],
                root / "hf-cache",
            ],
            timeout=7200,
            stdout_only=True,
            env=env,
        )
        model_path = Path(downloaded.stdout.strip().splitlines()[-1]).resolve()
        if model_path.name != p["model_revision"] or not model_path.is_relative_to(
            root
        ):
            raise RuntimeError("unexpected model snapshot location")
        validate_checkpoint(model_path)
        installed = {
            "state": "installed",
            "profile": p,
            "repo": str(ROOT),
            "gpu_index": args.gpu,
            "gpu": gpu,
            "port": args.port,
            "model_path": str(model_path),
            "engine_requirements": (HERE / "model-requirements.txt").read_text(),
        }
        installed["serve_command"] = serve_command(root, installed)
        (root / "supervisord.conf").write_text(supervisor_config(root))
        write_json(marker, installed)
        write_json(root / "install-state.json", {"state": "installed"})
        print(f"Installed; model not yet started: {root}")
    except BaseException as exc:
        write_json(
            root / "install-state.json",
            {"state": "failed", "error_type": type(exc).__name__},
        )
        raise


def control(root, *arguments, check=True):
    return subprocess.run(
        [
            str(root / "engine/bin/supervisorctl"),
            "-c",
            str(root / "supervisord.conf"),
            *arguments,
        ],
        check=check,
        capture_output=True,
        text=True,
        timeout=40,
    )


def endpoint(data):
    return f"http://127.0.0.1:{data['port']}"


def stop(root):
    result = control(root, "stop", PROGRAM, check=False)
    if result.returncode:
        status = control(root, "status", PROGRAM, check=False).stdout.split()
        if (
            len(status) < 2
            or status[0] != PROGRAM
            or status[1] not in {"STOPPED", "EXITED", "FATAL"}
        ):
            raise RuntimeError(
                "could not confirm model stopped; inspect supervisor.log"
            )
    write_json(root / "health.json", {"state": "stopped"})


def healthy_http(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + "/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def health(args, data):
    root = args.root
    # A previous ready result must not survive this check failing.
    write_json(root / "health.json", {"state": "checking"})
    try:
        status = control(root, "status", PROGRAM, check=False)
        if "RUNNING" not in status.stdout or not healthy_http(endpoint(data)):
            raise RuntimeError(
                "model process or HTTP health is not ready; inspect model.log"
            )
        run(
            [
                root / "bench/bin/python",
                "-m",
                "tools.vlm_benchmark.local_model_probe",
                "--endpoint",
                endpoint(data),
                "--model",
                data["profile"]["served_model"],
                "--output",
                root / "health.json",
            ],
            timeout=300,
            env=clean_environment(),
        )
        result = json.loads((root / "health.json").read_text())
        if result.get("state") != "ready":
            raise RuntimeError("vision/schema health checks failed")
        print(f"Ready: {data['profile']['served_model']} / GPU {data['gpu_index']}")
    except BaseException as exc:
        previous = json.loads((root / "health.json").read_text())
        if previous.get("state") != "failed":
            write_json(
                root / "health.json",
                {"state": "failed", "error_type": type(exc).__name__},
            )
        raise


def start(args, data):
    root = args.root
    engaged = False
    write_json(root / "health.json", {"state": "starting"})
    try:
        if control(root, "pid", check=False).returncode:
            run([root / "engine/bin/supervisord", "-c", root / "supervisord.conf"])
        status = control(root, "status", PROGRAM, check=False).stdout
        if "RUNNING" not in status:
            with socket.socket() as sock:
                try:
                    sock.bind(("127.0.0.1", data["port"]))
                except OSError:
                    raise RuntimeError(
                        "port occupied; refusing to probe another service"
                    ) from None
            engaged = True
            control(root, "start", PROGRAM)
        else:
            engaged = True
        until = time.monotonic() + args.wait_seconds
        while not healthy_http(endpoint(data)):
            status = control(root, "status", PROGRAM, check=False).stdout
            if "RUNNING" not in status or time.monotonic() >= until:
                raise RuntimeError(
                    "model exited or startup timed out; inspect model.log"
                )
            time.sleep(2)
        health(args, data)
    except BaseException as exc:
        if engaged:
            try:
                control(root, "stop", PROGRAM, check=False)
            except (OSError, subprocess.SubprocessError):
                pass
        previous = json.loads((root / "health.json").read_text())
        if previous.get("state") != "failed":
            write_json(
                root / "health.json",
                {"state": "failed", "error_type": type(exc).__name__},
            )
        raise


def benchmark_command(args, data):
    p = data["profile"]
    return [
        str(args.root / "bench/bin/python"),
        "-m",
        "tools.vlm_benchmark.modes",
        "--real",
        "--backend",
        "local",
        "--model",
        p["served_model"],
        "--endpoint",
        endpoint(data) + "/v1/chat/completions",
        "--metrics-url",
        endpoint(data) + "/metrics",
        "--model-revision",
        p["model_revision"],
        "--quantization",
        p["quantization"],
        "--engine-version",
        "vllm " + p["vllm_version"],
        "--cache-policy",
        p["cache_policy"],
        "--video",
        str(args.video.resolve()),
        "--streams",
        *map(str, args.streams),
        "--duration",
        str(args.duration),
        "--repeats",
        str(args.repeats),
        "--output",
        str(args.output.resolve()),
    ]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=[
            "plan",
            "package",
            "install",
            "start",
            "health",
            "status",
            "stop",
            "benchmark",
            "_serve",
        ],
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--wait-seconds", type=int, default=600)
    p.add_argument("--video", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--streams", nargs="+", type=int, default=[1, 4, 8, 12])
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--repeats", type=int, default=2)
    return p


def main():
    p = parser()
    args = p.parse_args()
    args.root = args.root.resolve()
    if (
        args.gpu < 0
        or not 1024 <= args.port <= 65535
        or not 1 <= args.wait_seconds <= 1800
        or len(str(args.root / "supervisor.sock").encode()) > 100
        or any(ch in str(args.root) for ch in "\r\n")
    ):
        p.error("invalid GPU/port/timeout or model-root path too long")
    if args.action == "plan":
        print(json.dumps(profile(), indent=2))
        return
    if args.action == "package":
        from tools.vlm_benchmark.run_remote import bundle

        if not args.output or not args.video or not args.video.is_file():
            p.error("package requires --video and a new --output .tar.gz")
        if args.output.exists():
            p.error("package output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        bundle(ROOT, [args.video], args.output, scope="local_gpu_model")
        print(args.output)
        return
    if platform.system() != "Linux":
        p.error("run this action on the Linux GPU server")
    if args.action == "_serve":
        data = load_install(args.root)
        env = {
            **clean_environment(),
            "CUDA_VISIBLE_DEVICES": str(data["gpu_index"]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        command = serve_command(args.root, data)
        os.execvpe(command[0], command, env)
    with operation_lock(args.root):
        if args.action == "install":
            install(args)
            return
        # Always allow stopping the old worker after code/profile updates.
        if args.action == "status":
            print(control(args.root, "status", PROGRAM, check=False).stdout)
            print(
                (args.root / "health.json").read_text()
                if (args.root / "health.json").exists()
                else "Not checked"
            )
            return
        if args.action == "stop":
            stop(args.root)
            return
        data = load_install(args.root)
        if args.action == "start":
            start(args, data)
        elif args.action == "health":
            health(args, data)
        elif args.action == "benchmark":
            if (
                not args.video
                or not args.video.is_file()
                or not args.output
                or args.output.exists()
                or not 3 <= args.duration <= 300
                or not 1 <= args.repeats <= 3
                or args.streams != sorted(set(args.streams))
                or any(not 1 <= n <= 12 for n in args.streams)
            ):
                p.error(
                    "benchmark requires a video, new output and valid load settings"
                )
            health(args, data)
            command = benchmark_command(args, data)
            # A failed/interrupting benchmark leaves partial reports via modes.py.
            try:
                result = subprocess.run(command, cwd=ROOT, env=clean_environment())
            finally:
                if args.output.exists():
                    shutil.copyfile(
                        args.root / "deployment.json", args.output / "deployment.json"
                    )
                    shutil.copyfile(
                        args.root / "health.json", args.output / "model-health.json"
                    )
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
