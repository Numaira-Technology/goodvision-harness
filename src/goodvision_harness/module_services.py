"""On-demand local workers, shared by camera runtimes on the same host.

Only fixed repository entry points can be launched. Remote/cloud modes never
start a local process. Workers stay warm until the owning application exits.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .workflow_modules import LOCAL_PORTS, MODELS, module_endpoint


def module_directory():
    return Path(
        os.environ.get("GOODVISION_MODULE_DATA_DIR", ".goodvision/modules")
    ).resolve()


@contextmanager
def worker_lock(path):
    """Hold a nonblocking OS lock for the worker's entire lifetime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            pass
        try:
            yield acquired
        finally:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)


class LocalModuleServices:
    def __init__(self):
        self._processes = {}
        self._lock = threading.RLock()

    @staticmethod
    def _check_engine(engine):
        if engine not in LOCAL_PORTS:
            raise ValueError("unknown local module")

    @staticmethod
    def _health(engine, endpoint):
        base = endpoint.rstrip("/")
        if engine == "bbox":
            parsed = urlsplit(base)
            base = f"{parsed.scheme}://{parsed.netloc}"
            path, headers = (
                "/health/tracking",
                {"x-goodvision-detector-model": "yolo26-s"},
            )
        else:
            base = base.removesuffix("/predict")
            path, headers = "/health", {}
        try:
            with httpx.Client(
                timeout=0.5, trust_env=False, follow_redirects=False
            ) as client:
                response = client.get(base + path, headers=headers)
                response.raise_for_status()
                value = response.json()
            ready = (
                isinstance(value, dict)
                and value.get("ready") is True
                and value.get("model")
                == ("yolo26-s" if engine == "bbox" else MODELS[engine])
                and (
                    engine != "bbox" or value.get("gpu_ready") is True
                    or (
                        value.get("device") == "cpu"
                        and value.get("runtime") == "ultralytics"
                        and value.get("provider_ready") is True
                        and value.get("gpu_ready") is False
                    )
                )
            )
            if ready:
                return "ready"
            if isinstance(value, dict) and (
                value.get("error") not in (None, "inference busy")
                or value.get("model")
                != ("yolo26-s" if engine == "bbox" else MODELS[engine])
                or (
                    engine == "bbox"
                    and value.get("gpu_ready") is False
                    and value.get("error") != "inference busy"
                )
            ):
                return "error"
            return "starting"
        except (httpx.HTTPStatusError, ValueError):
            # A listener answered, but it is unhealthy or not this module's API.
            # Loading another model cannot repair or replace its occupied port.
            return "error"
        except (httpx.ReadTimeout, httpx.ConnectTimeout):
            # A slow worker or a full connection backlog does not prove the port
            # is unused. Do not load another model while it may still be busy.
            return "starting"
        except httpx.ConnectError:
            return "unavailable"
        except httpx.HTTPError:
            return "error"

    def status(self, engine):
        self._check_engine(engine)
        endpoint = module_endpoint(engine, {"execution_mode": "local"})
        with self._lock:
            health = self._health(engine, endpoint)
            return self._status(engine, health)

    def _status(self, engine, health):
        if os.environ.get("GOODVISION_MODULE_MANAGER") == "compose":
            # Containers are owned by the host installer, including when their
            # URL happens to be loopback. Never create model processes or locks.
            return {
                "engine": engine,
                "state": "stopped" if health == "unavailable" else health,
                "ready": health == "ready", "manager": "compose",
            }
        if health != "unavailable":
            return {"engine": engine, "state": health, "ready": health == "ready"}
        process = self._processes.get(engine)
        if process is not None and process.poll() is None:
            state = "starting"
        else:
            with worker_lock(module_directory() / f"{engine}.lock") as available:
                state = "starting" if not available else "stopped"
            if state == "stopped" and process is not None:
                state = "error"
        return {"engine": engine, "state": state, "ready": False}

    def start(self, engine):
        self._check_engine(engine)
        with self._lock:
            endpoint = module_endpoint(engine, {"execution_mode": "local"})
            health = self._health(engine, endpoint)
            status = self._status(engine, health)
            if status.get("manager") == "compose":
                return status
            if health != "unavailable" or status["state"] == "starting":
                # Never launch over an existing listener, including an unhealthy
                # externally owned service. A failed child may be retried only
                # once its endpoint and process are both gone.
                return status
            parsed = urlsplit(endpoint)
            # A provisioned Docker/host worker can supply the local service URL.
            # Never try to start a process for a different machine or API path.
            default_path = "/v1/labels" if engine == "bbox" else ""
            if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or (
                parsed.path != default_path or parsed.scheme != "http"
            ):
                raise ValueError(
                    "The provisioned local module service is unavailable. "
                    "Start its worker on the deployment host."
                )
            directory = module_directory()
            directory.mkdir(parents=True, exist_ok=True)
            root = Path(__file__).resolve().parents[2]
            isolated = root / f".venv-module-{engine}" / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            python = os.environ.get(
                f"GOODVISION_{engine.upper()}_PYTHON",
                str(isolated) if isolated.is_file() else sys.executable,
            )
            command = [
                python,
                "-u",
                "-m",
                "goodvision_harness.module_service",
                engine,
                "--host",
                parsed.hostname,
                "--port",
                str(parsed.port or 80),
                "--data-dir",
                str(directory),
                "--singleton",
            ]
            environment = os.environ.copy()
            source = str(Path(__file__).resolve().parents[1])
            environment["PYTHONPATH"] = os.pathsep.join(
                [source, environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
            try:
                with (directory / f"{engine}.log").open("ab") as log:
                    self._processes[engine] = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        env=environment,
                        creationflags=subprocess.CREATE_NO_WINDOW
                        if os.name == "nt"
                        else 0,
                    )
            except OSError:
                raise ValueError(
                    "The local module could not start. Check its Python environment."
                ) from None
            return {"engine": engine, "state": "starting", "ready": False}

    def close(self):
        with self._lock:
            for process in self._processes.values():
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            self._processes.clear()


local_modules = LocalModuleServices()
atexit.register(local_modules.close)
