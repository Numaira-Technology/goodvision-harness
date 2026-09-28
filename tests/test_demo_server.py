from pathlib import Path

from goodvision_harness.demo.server import _parse_args


def test_demo_server_enables_private_persistent_console_settings_by_default() -> None:
    args = _parse_args([])

    assert args.deployment_settings == Path(".goodvision/deployment-settings.json")


def test_demo_server_accepts_an_explicit_console_settings_path() -> None:
    args = _parse_args(["--deployment-settings", "outputs/local-settings.json"])

    assert args.deployment_settings == Path("outputs/local-settings.json")
