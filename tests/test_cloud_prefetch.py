"""Remote prefetch wiring: credentials travel over stdin, never in argv."""

import json
import os
import subprocess
import sys

import pytest

from tools import cloud_prefetch as cp

R2_ENV = {
    "R2_ENDPOINT": "https://acc.r2.cloudflarestorage.com",
    "R2_BUCKET": "goodvision-models",
    "R2_ACCESS_KEY_ID": "akid-public",
    "R2_SECRET_ACCESS_KEY": "s3cr3t-value",
}
MODELS = [
    ("ZhuYuhan/VideoJev-4B", "1c1a253bf90dfa66176124d25a48ffb791d517d5"),
    ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
]


def test_credential_blob_includes_present_drops_empty():
    blob = cp.credential_blob({**R2_ENV, "R2_BUCKET": ""}, hf_token="hf_tok")
    assert "export R2_ENDPOINT=" in blob
    assert "export R2_ACCESS_KEY_ID=" in blob
    assert "export R2_SECRET_ACCESS_KEY=" in blob
    assert "R2_BUCKET" not in blob  # empty value dropped
    assert "export HF_TOKEN=" in blob


def test_credential_blob_without_hf_token():
    assert "HF_TOKEN" not in cp.credential_blob(R2_ENV)


def test_remote_command_sources_stdin_and_runs_prefetch():
    cmd = cp.remote_command("/w/cache", MODELS[0][0], MODELS[0][1], remote_dir="/w")
    assert ". /dev/stdin" in cmd
    assert "--repo ZhuYuhan/VideoJev-4B" in cmd
    assert "--revision 1c1a253bf90dfa66176124d25a48ffb791d517d5" in cmd
    assert "--cache-dir /w/cache" in cmd
    assert "/w/model_source.py" in cmd


def test_prefetch_models_carries_secret_only_on_stdin():
    calls = []
    uploads = []

    def runner(command, *, input=None, timeout=3600):
        calls.append((command, input))

    def uploader(ssh_options, host, port, remote_dir):
        uploads.append((host, port, remote_dir))

    cache = cp.prefetch_models(
        ssh_options=["-o", "BatchMode=yes"],
        host="1.2.3.4",
        port=2222,
        remote_dir="/workspace/models",
        models=MODELS,
        r2_env=R2_ENV,
        hf_token="hf_tok",
        runner=runner,
        uploader=uploader,
    )

    assert cache == "/workspace/models/cache"
    assert uploads == [("1.2.3.4", 2222, "/workspace/models")]
    assert len(calls) == len(MODELS)
    for (command, sent), (repo, revision) in zip(calls, MODELS, strict=True):
        # The secret is in the stdin blob...
        assert "s3cr3t-value" in sent
        # ...and never anywhere in the argv the process list would show.
        assert all("s3cr3t-value" not in part for part in command)
        assert command[0] == "ssh"
        assert "root@1.2.3.4" in command
        assert revision in command[-1]
        assert repo in command[-1]


def test_prefetch_defaults_cache_dir_under_remote_dir():
    seen = {}

    def runner(command, *, input=None, timeout=3600):
        seen["cmd"] = command

    cp.prefetch_models(
        ssh_options=[],
        host="h",
        port=22,
        remote_dir="/r",
        models=MODELS[:1],
        r2_env=R2_ENV,
        runner=runner,
        uploader=lambda *a: None,
    )
    assert "--cache-dir /r/cache" in seen["cmd"][-1]


@pytest.mark.skipif(os.name != "posix", reason="remote command targets a POSIX shell")
def test_remote_command_executes_quoted_paths_and_credentials(tmp_path):
    remote = tmp_path / "model's directory"
    remote.mkdir()
    (remote / "model_source.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps([sys.argv[1:], os.environ['R2_SECRET_ACCESS_KEY']]))\n"
    )
    cache = remote / "cache with spaces"
    secret = "quote' dollar$ semicolon; newline\nvalue"
    cmd = cp.remote_command(
        cache, *MODELS[0], python=sys.executable, remote_dir=str(remote)
    )
    result = subprocess.run(
        ["/bin/sh", "-c", cmd],
        input=cp.credential_blob({"R2_SECRET_ACCESS_KEY": secret}),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == [
        ["--repo", MODELS[0][0], "--revision", MODELS[0][1], "--cache-dir", str(cache)],
        secret,
    ]
    assert secret not in cmd
