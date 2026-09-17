"""Perception plugins and their isolated API mounts.

A plugin is a backend factory plus an immutable SessionProfile. Session lifetime,
input admission, decoding, source provenance and cancellation stay in one engine.
Removing a plugin removes one descriptor and its module, not shared branch trees.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace

if __package__:
    from .continuous import (
        SessionEngine,
        SessionProfile,
        SessionSettings,
        register_session_routes,
    )
else:
    from continuous import (
        SessionEngine,
        SessionProfile,
        SessionSettings,
        register_session_routes,
    )


def _enabled(name):
    value = os.environ.get(name, "false").lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def _sam_live(video, settings):
    if __package__:
        from .sam_live import SamLiveContinuation
    else:
        from sam_live import SamLiveContinuation
    return SamLiveContinuation(video, settings)


class DisabledBackend:
    gpu_ready = False

    def close(self):
        pass


@dataclass(frozen=True, slots=True)
class PerceptionPlugin:
    profile: SessionProfile
    name: str
    enable_env: str
    factory: Callable

    @property
    def api_prefix(self):
        return f"/v1/perception/{self.profile.route}"

    @property
    def health_path(self):
        return f"/health/perception/{self.profile.route}"


PLUGINS = (
    PerceptionPlugin(
        SessionProfile("sam_live", "sam3.1_video"),
        "SAM 连续分析",
        "SAM31_LIVE_ENABLED",
        _sam_live,
    ),
)


def build_perception_engines(video, registry, *, plugins=PLUGINS):
    engines = {}
    for plugin in plugins:
        settings = replace(
            SessionSettings.from_env(), enabled=_enabled(plugin.enable_env)
        )
        backend = (
            plugin.factory(video, settings) if settings.enabled else DisabledBackend()
        )
        engines[plugin.profile.route] = SessionEngine(
            video,
            settings,
            backend,
            profile=plugin.profile,
            production=True,
            registry=registry,
        )
    return engines


def register_perception_routes(
    app, engines, admission, decode_jpeg, *, plugins=PLUGINS
):
    app.state.perception_engines = engines
    for plugin in plugins:
        engine = engines[plugin.profile.route]
        register_session_routes(
            app,
            engine,
            admission,
            decode_jpeg,
            api_prefix=plugin.api_prefix,
            health_path=plugin.health_path,
            legacy=False,
        )

    @app.get("/v1/perception/routes")
    async def catalog():
        return {
            "contract_version": 1,
            "max_active_sessions": 1,
            "routes": [
                {
                    "id": plugin.profile.route,
                    "name": plugin.name,
                    "backend": plugin.profile.backend_name,
                    "enabled": engines[plugin.profile.route].settings.enabled,
                    "ready": engines[plugin.profile.route].gpu_ready,
                    "api_prefix": plugin.api_prefix,
                    "health_path": plugin.health_path,
                    "limits": engines[plugin.profile.route].limits(),
                    "error_code": engines[plugin.profile.route].error_code,
                }
                for plugin in plugins
            ],
        }
