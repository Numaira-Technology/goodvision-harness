"""Credential-free Docker configuration for read-only Compose rendering tests."""

import json
import shutil
from pathlib import Path


def isolated_docker_config(directory: Path) -> str:
    directory = Path(directory)
    binary = shutil.which("docker")
    if binary:
        plugins = Path(binary).resolve().parent.parent / "cli-plugins"
        if plugins.is_dir():
            (directory / "config.json").write_text(
                json.dumps({"cliPluginsExtraDirs": [str(plugins)]}), encoding="utf-8",
            )
    return str(directory)
