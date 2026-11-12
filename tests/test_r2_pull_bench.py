"""Offline checks for the R2 download A/B benchmark tool."""

import argparse

import pytest

from tools import r2_pull_bench as bench


def test_parse_variant_accepts_known_shapes():
    assert bench.parse_variant("threads:4x16") == ("threads", 4, 16)
    assert bench.parse_variant("procs:4x8") == ("procs", 4, 8)
    assert bench.parse_variant("s5cmd:16x16") == ("s5cmd", 16, 16)


@pytest.mark.parametrize("bad", ["threads", "gpu:4x4", "procs:4", "s5cmd:ax2", ""])
def test_parse_variant_rejects_garbage(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        bench.parse_variant(bad)


def test_main_requires_a_subcommand():
    with pytest.raises(SystemExit):
        bench.main([])
