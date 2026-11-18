"""Offline checks for the R2 download A/B benchmark tool."""

import argparse

import pytest

from tools import r2_pull_bench as bench


def test_parse_variant_accepts_known_shapes():
    assert bench.parse_variant("threads:4x16") == ("threads", 4, 16)
    assert bench.parse_variant("procs:4x8") == ("procs", 4, 8)
    assert bench.parse_variant("s5cmd:16x16") == ("s5cmd", 16, 16)
    assert bench.parse_variant("env:qwen8b") == ("env", "qwen8b", None)
    assert bench.parse_variant("envold:qwen8b") == ("envold", "qwen8b", None)
    assert bench.parse_variant("extractold:qwen8b") == ("extractold", "qwen8b", None)
    assert bench.parse_variant("both:qwen8b-vllm") == ("both", "qwen8b-vllm", None)


@pytest.mark.parametrize(
    "bad", ["threads", "gpu:4x4", "procs:4", "s5cmd:ax2", "", "env:", "env:a;b"]
)
def test_parse_variant_rejects_garbage(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_variant(bad)


def test_main_requires_a_subcommand():
    with pytest.raises(SystemExit):
        bench.main([])


class _Vast:
    def __init__(self, listings):
        self.listings = listings  # query -> offers

    def search_offers(self, query, **kw):
        return self.listings.get(query, [])


def _args(**kw):
    base = {"offer": 1, "max_dph": 0.12, "disk_gb": 80, "machine": None}
    return argparse.Namespace(**{**base, **kw})


def test_find_approved_offer_exact_id():
    vast = _Vast({"rentable=true dph_total<=0.12": [{"id": 1, "dph_total": 0.1}]})
    assert bench.find_approved_offer(vast, _args())["id"] == 1


def test_find_approved_offer_twin_slot_needs_machine():
    twin = {"id": 2, "dph_total": 0.1, "machine_id": 7, "num_gpus": 1}
    multi = {"id": 3, "dph_total": 0.05, "machine_id": 7, "num_gpus": 2}
    vast = _Vast({"machine_id=7": [twin, multi]})
    assert bench.find_approved_offer(vast, _args()) is None
    assert bench.find_approved_offer(vast, _args(machine=7))["id"] == 2


@pytest.mark.parametrize("kind", ["env", "envold", "extractold", "both"])
def test_env_benchmark_preserves_existing_prefix(tmp_path, monkeypatch, capsys, kind):
    from tools import env_source as es
    from tools import model_source as ms

    prefix = tmp_path / "existing"
    prefix.mkdir()
    sentinel = prefix / "keep"
    sentinel.write_text("existing environment")
    monkeypatch.setattr(
        es.registry, "env_of", lambda name: {"unpack_to": str(prefix), "size": 1}
    )
    monkeypatch.setattr(ms, "r2_client", lambda: None)
    monkeypatch.setattr(es, "pull_env", lambda *a, **kw: None)
    # The benchmark instruments these functions; keep each test isolated.
    for name in ("_sha256", "_safe_extract", "_run_unpack", "check_compatibility"):
        monkeypatch.setattr(es, name, getattr(es, name))
    monkeypatch.setattr(ms, "download_object", ms.download_object)
    monkeypatch.delitem(es.__dict__, "_bench_originals", raising=False)
    monkeypatch.delitem(ms.__dict__, "_bench_download_object", raising=False)
    bench._env_variant(kind, "synthetic", ms, _args(), tmp_path / "out")
    assert sentinel.read_text() == "existing environment"
    assert "FAILED FileExistsError" in capsys.readouterr().out


def test_env_benchmark_failure_preserves_competing_prefix(tmp_path, monkeypatch):
    from tools import env_source as es
    from tools import model_source as ms

    prefix = tmp_path / "competing"
    monkeypatch.setattr(
        es.registry, "env_of", lambda name: {"unpack_to": str(prefix), "size": 1}
    )
    monkeypatch.setattr(ms, "r2_client", lambda: None)

    def concurrent_restore(*a, **kw):
        prefix.mkdir()
        (prefix / "keep").write_text("another writer")
        raise FileExistsError("another restore won publication")

    monkeypatch.setattr(es, "pull_env", concurrent_restore)
    for name in ("_sha256", "_safe_extract", "_run_unpack", "check_compatibility"):
        monkeypatch.setattr(es, name, getattr(es, name))
    monkeypatch.setattr(ms, "download_object", ms.download_object)
    monkeypatch.delitem(es.__dict__, "_bench_originals", raising=False)
    monkeypatch.delitem(ms.__dict__, "_bench_download_object", raising=False)
    bench._env_variant("env", "synthetic", ms, _args(), tmp_path / "out")
    assert (prefix / "keep").read_text() == "another writer"


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-0.1"])
def test_invalid_price_cap_fails_before_credentials_or_rental(value):
    with pytest.raises(SystemExit) as error:
        bench.rent(["--offer", "1", "--max-dph=" + value, "--variants", "threads:4x8"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "value", ["procs:0x8", "threads:1x0", "procs:33x8", "threads:129x8", "s5cmd:4x129"]
)
def test_invalid_parallelism_is_not_silently_clamped(value):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_variant(value)


def test_remote_benchmark_quotes_all_arguments(tmp_path):
    import shlex

    args = _args(
        repo="model/with space; echo BAD",
        revision="sha$(echo BAD)",
        variants=[("procs", 4, 8), ("env", "qwen8b", None)],
    )
    remote = str(tmp_path / "directory with spaces")
    tokens = shlex.split(bench._box_command(args, remote))
    assert tokens == [
        "cd",
        remote,
        "&&",
        "python3",
        "-u",
        "r2_pull_bench.py",
        "box",
        "--repo",
        args.repo,
        "--revision",
        args.revision,
        "procs:4x8",
        "env:qwen8b",
    ]


def test_cleanup_reconciles_only_the_current_run_label():
    rows = [{"id": 1, "label": "owned"}, {"id": 2, "label": "other"}]
    vast = argparse.Namespace(show_instances=lambda: rows)
    released = []

    def release(client, instance):
        assert client is vast
        released.append(instance)
        rows[:] = [row for row in rows if row["id"] != instance]

    bench._release_owned(vast, None, "owned", release)
    assert released == [1]
    assert rows == [{"id": 2, "label": "other"}]


@pytest.mark.parametrize(
    "rows", [None, {}, [{"label": "owned"}], [{"id": 1, "label": "owned"}]]
)
def test_cleanup_requires_confirmed_instance_absence(rows):
    vast = argparse.Namespace(show_instances=lambda: rows)
    with pytest.raises(RuntimeError):
        bench._release_owned(vast, 1, "owned", lambda *a: None)


@pytest.mark.parametrize("kind", ["env", "envold", "extractold", "both"])
def test_unknown_env_is_rejected_before_rental(kind):
    with pytest.raises(SystemExit) as error:
        bench.rent(
            ["--offer", "1", "--max-dph", "0.1", "--variants", f"{kind}:no-such-model"]
        )
    assert error.value.code == 2
