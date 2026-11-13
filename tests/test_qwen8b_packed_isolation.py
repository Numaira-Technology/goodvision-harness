"""Exercise generated POSIX commands with synthetic modules and no GPU packages."""

import json
import os
import shutil
import subprocess
import venv

import pytest

from tools import diag_qwen8b_vllm_import as diagnostic
from tools import start_qwen8b_vllm as launcher

pytestmark = pytest.mark.skipif(os.name != "posix", reason="remote commands use bash")


@pytest.fixture
def isolated_runtime(tmp_path, monkeypatch):
    prefix = tmp_path / "empty-venv"
    # No pip, downloads, system site packages or real torch/vLLM imports.
    venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(
        prefix
    )
    python = str(prefix / "bin/python")
    external = tmp_path / "external"
    files = {
        "vllm/__init__.py": "__version__ = 'SYNTHETIC_PACKAGE'\n",
        "vllm/entrypoints/__init__.py": "",
        "vllm/entrypoints/openai/__init__.py": "",
        "vllm/entrypoints/openai/api_server.py": "print('SYNTHETIC_PACKAGE')\n",
        "torch/__init__.py": (
            "from types import SimpleNamespace\n"
            "__version__ = 'SYNTHETIC_PACKAGE'\n"
            "version = SimpleNamespace(cuda='synthetic')\n"
            "cuda = SimpleNamespace(is_available=lambda: True, "
            "get_device_name=lambda n: 'synthetic')\n"
        ),
        "torch/utils/__init__.py": "",
        "torch/utils/cpp_extension.py": "load_inline = None\n",
        "typing_extensions.py": "",
        "filelock.py": "",
        "tqdm.py": "",
    }
    for name, content in files.items():
        path = external / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    empty_cwd = tmp_path / "empty-cwd"
    empty_cwd.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    smi = fake_bin / "nvidia-smi"
    smi.write_text("#!/bin/sh\nprintf 'synthetic GPU status\\n'\n", encoding="utf-8")
    smi.chmod(0o700)
    captured = []
    with monkeypatch.context() as capture:
        capture.setattr(
            launcher.subprocess,
            "check_output",
            lambda argv, **kwargs: captured.append(argv[-1]) or "synthetic",
        )
        launcher._verify_restored_env(["unused-no-ssh"], python)
    commands = {
        "serve": launcher._vllm_serve_remote(python, "/synthetic/unused"),
        "verify": captured[0],
        "diagnostic": diagnostic._isolated_probe_remote(python, diagnostic.NEW_PROBE),
    }
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
    return python, external, empty_cwd, commands, env


@pytest.mark.parametrize(
    "command,location",
    [
        ("serve", "pythonpath"),
        ("verify", "pythonpath"),
        ("verify", "cwd"),
        ("diagnostic", "pythonpath"),
        ("diagnostic", "cwd"),
        ("serve", "prefix"),
        ("verify", "prefix"),
        ("diagnostic", "prefix"),
    ],
)
def test_generated_commands_use_only_packed_modules(isolated_runtime, command, location):
    python, external, empty_cwd, commands, env = isolated_runtime
    if location == "prefix":
        purelib = json.loads(
            subprocess.check_output(
                [python, "-I", "-c", "import json,sysconfig; "
                 "print(json.dumps(sysconfig.get_path('purelib')))"],
                env=env, text=True, timeout=15,
            )
        )
        shutil.copytree(external, purelib, dirs_exist_ok=True)
    elif location == "pythonpath":
        env["PYTHONPATH"] = str(external)
    result = subprocess.run(
        ["bash", "-c", commands[command]],
        cwd=external if location == "cwd" else empty_cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    if location == "prefix":
        assert result.returncode == 0, result.stderr
        assert "SYNTHETIC_PACKAGE" in result.stdout
    else:
        assert result.returncode != 0, result.stdout
        assert "No module named 'vllm'" in result.stderr
        assert "SYNTHETIC_PACKAGE" not in result.stdout
