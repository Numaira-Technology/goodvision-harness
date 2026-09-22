"""Compatibility launcher for the single unified Harness page."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from ..config import DEFAULT_CONFIG_PATH, load_config
from ..deployment_settings import DeploymentSettingsStore
from ..service import HarnessRuntime, create_app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--deployment-settings",
        type=Path,
        default=Path(".goodvision/deployment-settings.json"),
        help=(
            "persistent console settings "
            "(default: .goodvision/deployment-settings.json)"
        ),
    )
    parser.add_argument("--host", help="bind address (defaults to server.host)")
    parser.add_argument("--port", type=int, help="port (defaults to server.port)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config)
    application = create_app(
        lambda: HarnessRuntime(
            config,
            deployment_store=DeploymentSettingsStore(args.deployment_settings),
        )
    )
    uvicorn.run(
        application,
        host=args.host or config.server.host,
        port=args.port or config.server.port,
        log_level="info",
        server_header=False,
    )


if __name__ == "__main__":
    main()
