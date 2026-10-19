"""Disabled customer automatic-update entry points.

The retired package protocol is not replaced with unverified installation.
These compatibility entry points fail before reading packages, inspecting
Docker, stopping analysis, or modifying an installation.
"""

from __future__ import annotations

PRODUCT = "GoodVision Desktop Update"
DISABLED_MESSAGE = (
    "Customer automatic updates are disabled in this branch. "
    "No update package will be loaded or installed. Contact your maintainer."
)


class CustomerError(RuntimeError):
    """A short message suitable for the update window."""


def load_package(directory):
    raise CustomerError(DISABLED_MESSAGE)


def verify_archive(directory, manifest, progress):
    raise CustomerError(DISABLED_MESSAGE)


def find_projects(component, *, docker=None):
    raise CustomerError(DISABLED_MESSAGE)


def find_named_installations(component, name, *, docker=None):
    raise CustomerError(DISABLED_MESSAGE)


def verified_launcher_metadata(docker, deployment, manifest):
    raise CustomerError(DISABLED_MESSAGE)


def install(
    directory,
    project=None,
    progress=lambda _message: None,
    *,
    docker=None,
    selection=None,
):
    raise CustomerError(DISABLED_MESSAGE)
