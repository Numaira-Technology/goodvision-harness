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
