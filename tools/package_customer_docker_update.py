"""Disabled builder for the retired customer automatic-update package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .customer_update import CustomerError, DISABLED_MESSAGE
else:
    from customer_update import CustomerError, DISABLED_MESSAGE


def build_executable(output_directory):
    raise CustomerError(DISABLED_MESSAGE)


def add_config_ids(archive, images):
    raise CustomerError(DISABLED_MESSAGE)


def package(output, release, component, images, executable, *, docker=None):
    raise CustomerError(DISABLED_MESSAGE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--release")
    parser.add_argument("--component", choices=("harness", "contour"))
    parser.add_argument("--image", action="append", metavar="SERVICE=IMAGE")
    parser.add_argument("--executable", type=Path)
    parser.parse_args(argv)
    print(DISABLED_MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
