"""Explicit preparation/start operations for an already rented Linux container."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from . import profile as p


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def volume_root(path, mountinfo=None):
    root = Path(path).resolve(strict=True)
    if platform.system() != "Linux":
        raise ValueError("runtime requires Linux; plan/bundle/tests work offline")
    mounts = (
        Path("/proc/self/mountinfo").read_text() if mountinfo is None else mountinfo
    )
    mounted = {}
    for row in mounts.splitlines():
        before, _, after = row.partition(" - ")
        fields = before.split()
        if len(fields) >= 5 and after:
            name = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
            mounted[name] = after.split()[0]
    if (
        str(root) not in mounted
        or str(root) == "/"
        or mounted[str(root)]
        in {
            "tmpfs",
            "overlay",
        }
    ):
        raise ValueError("volume must be an explicit persistent mount, not root/tmpfs")
    # An OS mount proves a mount exists, not a paid Vast volume's lifecycle.
    marker = root / ".goodvision-vlm-volume.json"
    if not marker.exists():
        write_json(marker, {"purpose": "qwen35", "created": time.time()})
    elif json.loads(marker.read_text()).get("purpose") != "qwen35":
        raise ValueError("wrong volume marker")
    return root


@contextmanager
def exclusive(path):
    import fcntl

    with Path(path).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "another preparation/server operation owns this volume"
            ) from None
        yield


def model_dir(root, *, p=p):
    root = Path(root).resolve()
    target = root / "models" / p.MODEL_REVISION
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError("model directory must remain on the verified volume")
    return target


def verify_model(directory, *, p=p):
    directory = Path(directory)
    manifest = json.loads((directory / "verified.json").read_text())
    if (manifest["repo"], manifest["revision"]) != (p.MODEL_REPO, p.MODEL_REVISION):
        raise ValueError("model identity mismatch")
    files = manifest["files"]
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
    }
    if not required <= set(files):
        raise ValueError("model manifest is incomplete")
    for name, expected in files.items():
        path = directory / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory.resolve())
            or not path.is_file()
            or sha(path) != expected
        ):
            raise ValueError("model integrity check failed: " + name)
    actual_weights = {
        str(f.relative_to(directory)) for f in directory.rglob("*.safetensors")
    }
    if actual_weights != {name for name in files if name.endswith(".safetensors")}:
        raise ValueError("unexpected or missing weight shards")
    weights = json.loads((directory / "model.safetensors.index.json").read_text())
    if not set(weights["weight_map"].values()) <= set(files):
        raise ValueError("weight shard missing")
    if getattr(p, "QUANTIZATION", None) == "fp8":
        config = json.loads((directory / "config.json").read_text())
        if config.get("quantization_config", {}).get("quant_method") != "fp8":
            raise ValueError("expected prequantized FP8 weights")
    return manifest


def prepare(root, *, p=p):
    from huggingface_hub import HfApi, snapshot_download

    root = volume_root(root)
    started = time.monotonic()

    def record(manifest, reused):
        logs = root / "runtime"
        logs.mkdir(exist_ok=True)
        write_json(
            logs / "last-preparation.json",
            {
                "revision": p.MODEL_REVISION,
                "reused_verified_weights": reused,
                "elapsed_seconds": time.monotonic() - started,
                "finished_at_unix": time.time(),
            },
        )
        return manifest

    with exclusive(root / ".model.lock"):
        target = model_dir(root, p=p)
        if (target / "verified.json").exists():
            return record(verify_model(target, p=p), True)
        # Resume downloads, but never treat mere directory existence as ready.
        target.mkdir(parents=True, exist_ok=True)
        info = HfApi().model_info(
            p.MODEL_REPO, revision=p.MODEL_REVISION, files_metadata=True
        )
        if info.sha != p.MODEL_REVISION:
            raise ValueError("hub returned a different model revision")
        selected = [
            f
            for f in info.siblings
            if Path(f.rfilename).suffix
            in {".json", ".safetensors", ".jinja", ".txt", ".model"}
        ]
        # Cache fragments and unrelated files do not count as usable weights.
        required_space = 0
        for file in selected:
            if type(file.size) is not int or file.size < 0:
                raise ValueError("model metadata is missing file sizes")
            path = target / file.rfilename
            if path.is_symlink() or not path.resolve().is_relative_to(target.resolve()):
                raise ValueError("unsafe model path")
            reusable = path.is_file() and path.stat().st_size == file.size
            if reusable:
                if file.lfs:
                    reusable = sha(path) == file.lfs.sha256
                else:
                    data = path.read_bytes()
                    reusable = (
                        hashlib.sha1(
                            b"blob " + str(len(data)).encode() + b"\0" + data
                        ).hexdigest()
                        == file.blob_id
                    )
            if not reusable:
                required_space += file.size
        if shutil.disk_usage(root).free < required_space + 10 * 1024**3:
            raise ValueError(
                "insufficient volume space for weights plus working margin"
            )
        snapshot_download(
            p.MODEL_REPO,
            revision=p.MODEL_REVISION,
            local_dir=target,
            allow_patterns=[f.rfilename for f in selected],
        )
        hashes = {}
        for file in selected:
            path = target / file.rfilename
            if not path.resolve().is_relative_to(target.resolve()) or path.is_symlink():
                raise ValueError("unsafe model path")
            if path.stat().st_size != file.size:
                raise ValueError("downloaded model file size differs")
            digest = sha(path)
            if file.lfs and digest != file.lfs.sha256:
                raise ValueError("downloaded weight checksum differs")
            if not file.lfs:
                content = path.read_bytes()
                blob = b"blob " + str(len(content)).encode() + b"\0" + content
                if hashlib.sha1(blob).hexdigest() != file.blob_id:
                    raise ValueError("downloaded config checksum differs")
            hashes[file.rfilename] = digest
        manifest = {"repo": p.MODEL_REPO, "revision": p.MODEL_REVISION, "files": hashes}
        write_json(target / "verified.json", manifest)
        return record(verify_model(target, p=p), False)


def wait_for_server(process, timeout=900):
    """Measure actual load/readiness time without sending inference requests."""
    started = time.monotonic()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited before readiness; inspect server.log")
        try:
            with opener.open(p.BASE_URL + "/health", timeout=2) as response:
                if response.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("vLLM startup exceeded 900 seconds")


def serve_command(root, *, p=p):
    return (
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_dir(root, p=p)),
            "--served-model-name",
            p.SERVED_MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            str(getattr(p, "SERVER_SEQUENCES", 48)),
            "--gpu-memory-utilization",
            str(getattr(p, "GPU_UTILIZATION", 0.90)),
            "--limit-mm-per-prompt",
            '{"image":4,"video":0}',
            "--no-enable-prefix-caching",
            "--mm-processor-cache-gb",
            "0",
            "--default-chat-template-kwargs",
            '{"enable_thinking":false}',
            "--no-enable-log-requests",
        ]
        + (
            ["--quantization", "fp8"]
            if getattr(p, "QUANTIZATION", None) == "fp8"
            else []
        )
        + list(getattr(p, "SERVER_OPTIONS", ()))
    )


def start(root, *, p=p):
    root = volume_root(root)
    requested = time.monotonic()
    with exclusive(root / ".server.lock"), exclusive(root / ".model.lock"):
        verify_model(model_dir(root, p=p), p=p)
        if importlib.metadata.version("vllm") != p.ENGINE_VERSION:
            raise ValueError("vLLM version differs from frozen image")
        import torch

        if torch.cuda.device_count() != 1:
            raise ValueError("baseline requires one dedicated GPU")
        gpu = torch.cuda.get_device_properties(0)
        if hasattr(p, "validate_gpu"):
            p.validate_gpu(gpu)
        elif gpu.total_memory < 110 * 1024**3:
            raise ValueError(
                "BF16 baseline requires >=110 GiB GPU memory; no CPU offload"
            )
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", 8000))
            except OSError:
                raise RuntimeError(
                    "port 8000 is already occupied; do not reuse an unknown server"
                ) from None
        logs = root / "runtime"
        logs.mkdir(exist_ok=True)
        run = logs / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run.mkdir()  # Never overwrite the previous startup record.
        command = serve_command(root, p=p)
        extra = p.deployment_extra(root) if hasattr(p, "deployment_extra") else {}
        write_json(
            run / "deployment.json",
            {
                "version": p.VERSION,
                "model": p.MODEL_REPO,
                "revision": p.MODEL_REVISION,
                "image_expected": p.IMAGE,
                "vllm_version": p.ENGINE_VERSION,
                "gpu": gpu.name,
                "vram_bytes": gpu.total_memory,
                "command": command,
                **extra,
                "image_attestation": "operator must use pinned Vast template",
            },
        )
        env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
        env.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            VLLM_CACHE_ROOT=str(
                root / "cache" / p.ENGINE_VERSION / gpu.name.replace(" ", "_")
            ),
        )
        if hasattr(p, "server_environment"):
            env.update(p.server_environment(root))
        # No cloud credentials are necessary for serving already verified weights.
        for key in list(env):
            if any(term in key.upper() for term in ("API_KEY", "HF_TOKEN")):
                env.pop(key)
        with (run / "server.log").open("w") as log:

            def terminate(_signum, _frame):
                raise KeyboardInterrupt

            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=log, start_new_session=True
            )
            previous = signal.signal(signal.SIGTERM, terminate)
            try:
                write_json(
                    root / "runtime/active.json",
                    {
                        "pid": process.pid,
                        "start_ticks": process_start_ticks(process.pid),
                        "boot_id": boot_id(),
                        "version": p.VERSION,
                        "revision": p.MODEL_REVISION,
                        "deployment": str(run / "deployment.json"),
                    },
                )
                ready_seconds = wait_for_server(process)
                if hasattr(p, "attest_ready"):
                    p.attest_ready(run)
                write_json(
                    run / "startup-timing.json",
                    {
                        "process_to_health_seconds": ready_seconds,
                        "including_weight_verification_seconds": time.monotonic()
                        - requested,
                        "scope": "health ready, not inference warmup or capacity",
                    },
                )
                returncode = process.wait()
            finally:
                signal.signal(signal.SIGTERM, previous)
                try:
                    stop_process_group(process)
                finally:
                    (root / "runtime/active.json").unlink(missing_ok=True)
            if returncode:
                raise RuntimeError("vLLM exited; inspect " + str(run / "server.log"))


def process_start_ticks(pid):
    # comm can include spaces, so split after its final closing parenthesis.
    fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        raise ValueError("server process has exited")
    return fields[19]


def verify_active(root, *, p=p):
    root = volume_root(root)
    active = json.loads((root / "runtime/active.json").read_text())
    if (
        active["version"] != p.VERSION
        or active["revision"] != p.MODEL_REVISION
        or active.get("boot_id") != boot_id()
        or process_start_ticks(active["pid"]) != active["start_ticks"]
    ):
        raise ValueError("no matching live self-hosted server")
    if hasattr(p, "verify_deployment"):
        p.verify_deployment(root, active)
    return active


def boot_id():
    """A persistent volume can outlive the host boot and reuse a PID/tick pair."""
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def stop_process_group(process, timeout=30):
    """Only stop the private session created for this service, including workers."""

    def send(sig):
        try:
            os.killpg(process.pid, sig)
            return True
        except ProcessLookupError:
            return False

    send(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    # The API parent may exit before its GPU workers; still clean that session.
    send(signal.SIGKILL)
    process.wait()
