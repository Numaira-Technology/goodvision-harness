"""Read tools/models.yaml -- the single registry of pinned models in R2.

Each model has up to two keys:

  weights  -- repo + revision (+ the R2 prefix the weights live under)
  env      -- a pre-built, conda-pack'd environment tarball in R2, or None

This is a thin reader with no side effects. ``tools/model_source.py`` fetches
weights; ``tools/env_source.py`` fetches and activates the env; this module just
tells them where to look.
"""

import os
from pathlib import Path

REGISTRY_PATH = Path(__file__).with_name("models.yaml")


class RegistryError(RuntimeError):
    pass


def _load(path=None):
    import yaml  # PyYAML, already a harness dependency

    p = Path(path) if path else REGISTRY_PATH
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise RegistryError(f"cannot read registry {p}: {e}") from e
    if not isinstance(data.get("models"), dict):
        raise RegistryError(f"registry {p} has no 'models' mapping")
    return data


def model_names(path=None):
    return sorted(_load(path)["models"])


def get_model(name, path=None):
    """Return the raw registry entry for ``name`` (dict), or raise."""
    models = _load(path)["models"]
    if name not in models:
        raise RegistryError(f"unknown model {name!r}; known: {sorted(models)}")
    return models[name]


def weights_of(name, path=None):
    """(repo, revision) for ``name`` -- what model_source.ensure_cached wants."""
    entry = get_model(name, path)
    repo, rev = entry.get("repo"), entry.get("revision")
    if not repo or not rev:
        raise RegistryError(f"model {name!r} is missing repo/revision")
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
        raise RegistryError(f"model {name!r} revision is not a 40-char sha")
    return repo, rev


def env_of(name, path=None):
    """The ``env`` spec for ``name`` (dict), or None when there is no packed env."""
    entry = get_model(name, path)
    env = entry.get("env")
    if env is None:
        return None
    for req in ("r2_key", "unpack_to", "activate_rel"):
        if not env.get(req):
            raise RegistryError(f"model {name!r} env is missing {req!r}")
    return env


def bucket(path=None):
    """Registry default bucket, overridden by R2_BUCKET when set."""
    return os.environ.get("R2_BUCKET") or _load(path).get("defaults", {}).get("bucket")
