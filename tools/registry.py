"""Read and validate the separate, pinned weight and environment identities."""

import os
import re
from pathlib import Path, PurePosixPath

REGISTRY_PATH = Path(__file__).with_name("models.yaml")
GPU_ARCHES = {"ada": ("8.9",), "ampere": ("8.0", "8.6"), "hopper": ("9.0",)}


class RegistryError(ValueError):
    pass


def _load(path=None):
    import yaml

    p = Path(path) if path else REGISTRY_PATH
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise RegistryError(f"cannot read registry {p}") from e
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data["version"] != 1
    ):
        raise RegistryError("registry must be a version 1 mapping")
    if not isinstance(data.get("models"), dict) or not all(
        isinstance(k, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]*", k)
        for k in data["models"]
    ):
        raise RegistryError("registry must have a named 'models' mapping")
    return data


def model_names(path=None):
    return sorted(_load(path)["models"])


def get_model(name, path=None):
    models = _load(path)["models"]
    if name not in models:
        raise RegistryError(f"unknown model {name!r}; known: {sorted(models)}")
    entry = models[name]
    if not isinstance(entry, dict):
        raise RegistryError(f"model {name!r} must be a mapping")
    return entry


def weights_of(name, path=None):
    """(repo, revision) for model_source.ensure_cached; never an env object key."""
    entry = get_model(name, path)
    repo, rev = entry.get("repo"), entry.get("revision")
    if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise RegistryError(f"model {name!r} repo must be org/name")
    if any(part in {".", ".."} for part in repo.split("/")):
        raise RegistryError("invalid model repo")
    if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise RegistryError(f"model {name!r} revision is not a 40-char sha")
    expected = f"models/{repo}/{rev}/"
    weights = entry.get("weights")
    if not isinstance(weights, dict) or weights.get("r2_prefix") != expected:
        raise RegistryError("weight prefix must match the pinned repo/revision")
    return repo, rev


def relative_path(value):
    """Validate portable archive paths and object keys without normalizing escapes."""
    if (
        not isinstance(value, str)
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(c in value for c in "\\:\x00\r\n")
    ):
        raise RegistryError("expected a safe relative path")
    return PurePosixPath(value)


def validate_env(env):
    """Require a trusted pin and complete metadata before any filesystem/network IO."""
    if not isinstance(env, dict):
        raise RegistryError("env must be a mapping")
    for key, suffix in (("r2_key", ".tar.gz"), ("manifest_key", ".json")):
        path = relative_path(env.get(key))
        if (
            path.parts[0] != "envs"
            or len(path.parts) < 3
            or not str(path).endswith(suffix)
        ):
            raise RegistryError(
                f"{key} must identify an envs/ artifact ending in {suffix}"
            )
    if env["manifest_key"] != env["r2_key"][:-7] + ".json":
        raise RegistryError("env tarball and manifest keys must share an identity")
    sha = env.get("sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RegistryError("env sha256 must be a pinned 64-character lowercase digest")
    if type(env.get("size")) is not int or env["size"] <= 0:
        raise RegistryError("env size must be a positive integer")
    prefix = env.get("unpack_to")
    if not isinstance(prefix, str) or not prefix.startswith("/"):
        raise RegistryError("unpack_to must be an absolute POSIX prefix")
    if len(relative_path(prefix[1:]).parts) < 2:
        raise RegistryError("unpack_to must name a dedicated environment directory")
    relative_path(env.get("activate_rel"))
    if env["activate_rel"] != "bin/activate":
        raise RegistryError("conda-pack activate_rel must be bin/activate")
    compat = env.get("compat")
    if not isinstance(compat, dict):
        raise RegistryError("env compat must be a mapping")
    for key in ("python", "cuda", "torch"):
        if not isinstance(compat.get(key), str) or not re.fullmatch(
            r"\d+\.\d+(?:\.\d+)?", compat[key]
        ):
            raise RegistryError(f"compat.{key} must be a numeric version string")
    if (
        not isinstance(compat.get("gpu_family"), str)
        or compat["gpu_family"] not in GPU_ARCHES
    ):
        raise RegistryError("unsupported compat.gpu_family")
    return env


def env_of(name, path=None):
    entry = get_model(name, path)
    env = entry.get("env")
    return None if env is None else validate_env(env)


def bucket(path=None):
    """Registry default bucket, overridden by R2_BUCKET when set."""
    return os.environ.get("R2_BUCKET") or _load(path).get("defaults", {}).get("bucket")
