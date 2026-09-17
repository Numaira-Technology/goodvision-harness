"""Compatibility launcher for the single unified Harness page."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from ..config import DEFAULT_CONFIG_PATH, load_config
from ..deployment_settings import DeploymentSettingsStore
from ..service import HarnessRuntime, create_app

_LOCAL_DEPLOYMENT_SETTINGS = Path(".goodvision") / "deployment-settings.json"


def local_deployment_store() -> DeploymentSettingsStore:
    """Keep console API-key saves working for native Windows/macOS demos."""

    path = os.environ.get("HARNESS_DEPLOYMENT_SETTINGS_PATH", "").strip()
    if not path:
        path = str(_LOCAL_DEPLOYMENT_SETTINGS)
    return DeploymentSettingsStore(path)


def create_demo_app(config=None):
    loaded = config if config is not None else load_config(DEFAULT_CONFIG_PATH)
    store = local_deployment_store()
    return create_app(lambda: HarnessRuntime(loaded, deployment_store=store))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", help="bind address (defaults to server.host)")
    parser.add_argument("--port", type=int, help="port (defaults to server.port)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config)
    application = create_demo_app(config)
    uvicorn.run(
        application,
        host=args.host or config.server.host,
        port=args.port or config.server.port,
        log_level="info",
        server_header=False,
    )


if __name__ == "__main__":
    main()
