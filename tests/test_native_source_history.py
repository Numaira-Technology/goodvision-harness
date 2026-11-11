"""Exercise pinned source preparation against a local shallow Git repository."""
import json
from pathlib import Path
import subprocess

import pytest

from tools.unified_benchmark import fetch_native_sources as history
from tools.unified_benchmark import native_deploy as deploy
from tools.unified_benchmark.contract import file_hash

ROOT = Path(__file__).resolve().parents[1]


def git(repository, *args, check=True):
    return subprocess.run(["git", "-C", str(repository), *args],
                          check=check, capture_output=True, text=True)


def test_prepare_shallow_history_then_verify_every_native_source(tmp_path):
    repository = tmp_path / "shallow"
    repository.mkdir()
    git(repository, "init", "-q")
    # Origin is this local source checkout, never GitHub/R2/a model repository.
    git(repository, "remote", "add", "origin", str(ROOT))
    git(repository, "fetch", "--depth=1", "origin", "HEAD")
    git(repository, "checkout", "--detach", "FETCH_HEAD")
    original_head = git(repository, "rev-parse", "HEAD").stdout.strip()
    sources = json.loads(history.SOURCES.read_text())
    pins = {spec["commit"] for spec in sources.values()}
    for pin in pins:
        assert git(repository, "cat-file", "-e", pin, check=False).returncode != 0

    assert set(history.fetch_sources(repository)) == pins
    assert git(repository, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
    assert git(repository, "rev-parse", "HEAD").stdout.strip() == original_head
    for alias in ("videojev", "decision3"):
        exported = tmp_path / alias
        hashes = deploy.export_service(alias, exported, repository=repository)
        assert len(hashes) > 100
        assert hashes == sources[alias]["files"]
        assert all(file_hash(exported / name) == sha for name, sha in hashes.items())


@pytest.mark.parametrize("pin", ["main", "dcea006", "--upload-pack=unexpected", None, 123])
def test_reject_nonimmutable_source_pin_before_fetch(tmp_path, monkeypatch, pin):
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"synthetic": {"commit": pin}}))
    monkeypatch.setattr(history, "SOURCES", manifest)
    monkeypatch.setattr(history.subprocess, "run", lambda *a, **k: pytest.fail("invalid pin fetched"))
    with pytest.raises(ValueError, match="full commit SHA-1"):
        history.fetch_sources(tmp_path)
