from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_windows_hybrid_does_not_build_start_or_wait_for_cpu_rich_worker() -> None:
    script = (ROOT / "tools" / "start_windows_hybrid_demo.ps1").read_text(
        encoding="utf-8"
    )

    assert 'RICH_LABEL_ENABLED = "false"' in script
    assert "compose.rich-cpu-host.yaml" not in script
    assert "& docker" not in script
    assert "Get-Command docker" not in script
    assert "Start-Sleep" not in script
    assert "RichReadyTimeoutSeconds" not in script


def test_windows_hybrid_script_requires_directml_and_full_face_schedule() -> None:
    script = (ROOT / "tools" / "start_windows_hybrid_demo.ps1").read_text(
        encoding="utf-8"
    )

    assert 'FACE_PROVIDER = "DmlExecutionProvider"' in script
    assert 'FACE_LIVE_SCHEDULER = "full"' in script
    assert 'RICH_LABEL_ENABLED = "true"' in script
    assert 'HARNESS_DRY_RUN = "false"' in script
    assert 'HARNESS_AUTOSTART = "false"' in script
    assert "DmlExecutionProvider is unavailable" in script
    assert '$gpuProperty.Value -is [bool]' in script
    assert 'Invoke-RestMethod -Uri $richReadyUrl -TimeoutSec 3' in script


def test_windows_hybrid_instructions_keep_runtime_packages_isolated() -> None:
    instructions = (ROOT / "WINDOWS_HYBRID.md").read_text(encoding="utf-8")

    assert ".venv-directml" in instructions
    assert ".[dev,face-directml,webrtc]" in instructions
    assert "onnxruntime-directml" in instructions
    assert "DmlExecutionProvider" in instructions
    assert "FACE_LIVE_SCHEDULER=full" in instructions
    assert "RICH_LABEL_ENABLED=false" in instructions
    assert "Docker Desktop is not required" in instructions
    assert "gpu_ready: true" in instructions
    assert "compose.rich-cpu-host.yaml" not in instructions


def _ps_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@pytest.mark.parametrize(
    ("health_payload", "endpoint_supplied", "expected_enabled"),
    [
        (None, False, False),
        ({"ready": True, "gpu_ready": True}, True, True),
        ({"ready": True, "gpu_ready": False}, True, False),
        ({"ready": True, "gpu_ready": "true"}, True, False),
        ({"ready": True}, True, False),
        (None, True, False),
    ],
)
def test_launcher_requires_verified_gpu_and_preserves_face_event_startup(
    tmp_path: Path,
    health_payload: dict | None,
    endpoint_supplied: bool,
    expected_enabled: bool,
) -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required to execute the Windows launcher")
    capture = tmp_path / "harness-environment.json"
    restored = tmp_path / "restored-environment.json"
    fake_python = tmp_path / "python.ps1"
    fake_python.write_text(
        """if ($args[0] -eq '-c') {
    Write-Output '["DmlExecutionProvider"]'
}
else {
    [pscustomobject]@{
        rich = $env:RICH_LABEL_ENABLED
        endpoint = $env:RICH_LABEL_ENDPOINT
        face = $env:FACE_PROVIDER
        schedule = $env:FACE_LIVE_SCHEDULER
        dry_run = $env:HARNESS_DRY_RUN
        autostart = $env:HARNESS_AUTOSTART
        arguments = @($args)
    } | ConvertTo-Json | Set-Content -LiteralPath $env:HYBRID_TEST_CAPTURE
}
$global:LASTEXITCODE = 0
""",
        encoding="utf-8",
    )
    probe_result = (
        "throw 'worker unavailable'" if health_payload is None
        else f"return ({_ps_literal(json.dumps(health_payload))} | ConvertFrom-Json)"
    )
    endpoint = "http://127.0.0.1:8091/v1/labels"
    endpoint_argument = (
        f" -RichEndpoint {_ps_literal(endpoint)}" if endpoint_supplied else ""
    )
    launcher = ROOT / "tools" / "start_windows_hybrid_demo.ps1"
    wrapper = tmp_path / "launch-test.ps1"
    wrapper.write_text(
        f"""$ErrorActionPreference = 'Stop'
$global:probeCalls = 0
function docker {{ throw 'The launcher must not start a container' }}
function Invoke-RestMethod {{
    param($Uri, $TimeoutSec)
    $global:probeCalls += 1
    if ($Uri -ne 'http://127.0.0.1:8091/health/ready' -or $TimeoutSec -ne 3) {{
        throw 'unexpected health probe'
    }}
    {probe_result}
}}
$env:HYBRID_TEST_CAPTURE = {_ps_literal(capture)}
$env:RICH_LABEL_ENABLED = 'original-enabled'
$env:RICH_LABEL_ENDPOINT = 'original-endpoint'
& {_ps_literal(launcher)} -PythonPath {_ps_literal(fake_python)}{endpoint_argument}
[pscustomobject]@{{
    rich = $env:RICH_LABEL_ENABLED
    endpoint = $env:RICH_LABEL_ENDPOINT
    probes = $global:probeCalls
}} | ConvertTo-Json | Set-Content -LiteralPath {_ps_literal(restored)}
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(wrapper)],
        capture_output=True, text=True, timeout=20, check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    observed = json.loads(capture.read_text(encoding="utf-8-sig"))
    assert observed["rich"] == ("true" if expected_enabled else "false")
    if expected_enabled:
        assert observed["endpoint"] == endpoint
    else:
        assert not observed["endpoint"]
    assert observed["face"] == "DmlExecutionProvider"
    assert observed["schedule"] == "full"
    assert observed["dry_run"] == observed["autostart"] == "false"
    assert observed["arguments"] == ["-m", "goodvision_harness.service"]
    previous = json.loads(restored.read_text(encoding="utf-8-sig"))
    assert previous == {
        "rich": "original-enabled", "endpoint": "original-endpoint",
        "probes": int(endpoint_supplied),
    }
