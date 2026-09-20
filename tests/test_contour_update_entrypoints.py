"""Execute the shipped Windows entry points in fresh PowerShell processes.

A native Docker command double keeps these regressions independent of a daemon.
Unlike dot-sourcing, -File exercises advanced-script parameter default binding.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile

import pytest

from workers.contour.package_update import BASE_IMAGES, package

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows entry points")


@pytest.fixture
def windows_update(tmp_path):
    archive = tmp_path / "update.zip"
    package(archive)
    extracted = tmp_path / "package with spaces [update]"
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    bundle = next(extracted.iterdir())
    node = tmp_path / "installed node [keep]"
    node.mkdir()
    for name in (".env", "compose.yaml", "compose.gpu.yaml", "model", "key"):
        (node / name).write_text("preserve original", encoding="utf-8")
    commands = tmp_path / "fake docker"
    commands.mkdir()

    def response(name, value):
        (commands / name).write_text(json.dumps(value), encoding="ascii")

    response(
        "context.json",
        [{"Name": "desktop-linux", "Endpoints": {"docker": {"Host": "npipe:fixture"}}}],
    )
    response("info.json", {"OSType": "linux", "Architecture": "amd64"})
    response(
        "config.json",
        {
            "services": {
                item["service"]: {"image": item["image"]} for item in BASE_IMAGES
            }
        },
    )
    response(
        "status.json",
        {
            "gpu_ready": True,
            "active_sessions": 0,
            "accepting_sessions": True,
            "requests": {"in_flight": 0},
            "available": True,
            "reclaim_count": 1,
            "error_code": None,
            "allocated_bytes": 100,
            "unused_reserved_bytes": 0,
            "device_free_bytes": 900,
        },
    )
    for item in BASE_IMAGES:
        response(
            item["service"] + ".json",
            [
                {
                    "Image": item["base_id"],
                    "Config": {
                        "Labels": {
                            "com.docker.compose.project.working_dir": str(node),
                            "com.docker.compose.project.config_files": str(
                                node / "compose.yaml"
                            )
                            + ","
                            + str(node / "compose.gpu.yaml"),
                        }
                    },
                    "State": {"Status": "running", "OOMKilled": False},
                }
            ],
        )
    engine, gateway = BASE_IMAGES
    stub = r"""@echo off
>>"%~dp0calls.log" echo %*
if "%~1"=="context" goto context
if "%~1"=="info" goto info
if "%~1"=="ps" goto ps
if "%~1"=="inspect" goto inspect
if "%~1"=="image" goto image
if "%~1"=="compose" goto compose
if "%~1"=="build" exit /b 0
if "%~1"=="tag" exit /b 0
exit /b 91
:context
type "%~dp0context.json"
exit /b 0
:info
type "%~dp0info.json"
exit /b 0
:ps
echo %* | findstr /C:"service=engine" >nul
if not errorlevel 1 (echo engine) else (echo contour)
exit /b 0
:inspect
type "%~dp0%~2.json"
exit /b 0
:image
if "%~5"=="goodvision-contour-runtime:0.1.0" (echo ENGINE_ID) else (echo GATEWAY_ID)
exit /b 0
:compose
echo %* | findstr /C:" config " >nul
if not errorlevel 1 goto config
echo %* | findstr /C:" exec " >nul
if not errorlevel 1 goto status
exit /b 0
:config
type "%~dp0config.json"
exit /b 0
:status
type "%~dp0status.json"
exit /b 0
"""
    stub = stub.replace("ENGINE_ID", engine["base_id"]).replace(
        "GATEWAY_ID", gateway["base_id"]
    )
    (commands / "docker.cmd").write_text(stub, encoding="ascii")
    elsewhere = tmp_path / "unrelated working directory"
    elsewhere.mkdir()
    environment = os.environ | {"PATH": str(commands) + os.pathsep + os.environ["PATH"]}
    # A fresh Explorer launch does not inherit PowerShell 7's module path.
    for name in list(environment):
        if name.upper() in ("DOCKER_HOST", "DOCKER_CONTEXT", "PSMODULEPATH"):
            environment.pop(name)
    return bundle, node, commands, elsewhere, environment


@pytest.mark.parametrize("entry", ["script_default", "script_explicit", "launcher"])
def test_shipped_entry_point_resolves_package_outside_current_directory(
    windows_update, entry
):
    bundle, node, commands, elsewhere, environment = windows_update
    if entry == "launcher":
        invocation = ["cmd.exe", "/d", "/c", str(bundle / "UPDATE.cmd")]
    else:
        invocation = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(bundle / "update_contour.ps1"),
            "-CheckOnly",
        ]
        if entry == "script_explicit":
            invocation += ["-PackageDir", str(bundle)]
    result = subprocess.run(
        invocation,
        cwd=elsewhere,
        env=environment,
        input="\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout
    assert "empty string" not in result.stdout
    assert (
        "Update ready." if entry == "launcher" else "checks passed"
    ) in result.stdout
    calls = (commands / "calls.log").read_text(encoding="utf-8")
    assert "context inspect" in calls
    assert "config --format json" in calls
    if entry != "launcher":
        assert "build " not in calls and "stop " not in calls and "tag " not in calls
    for name in (".env", "compose.yaml", "compose.gpu.yaml", "model", "key"):
        assert (node / name).read_text(encoding="utf-8") == "preserve original"
