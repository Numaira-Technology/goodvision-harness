"""Build and verify a GPU or explicitly CPU-built wheel environment.

No weights are built or uploaded. Existing prefixes, packs and R2 objects are
never replaced. The output is a proposed registry pin; review it before use.
A trusted installed Miniforge, or a local installer plus its independently
verified SHA256, is required. Nothing is fetched from a floating installer URL.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from tools import env_source, model_source, registry

# R2 documents create-once PutObject; do not assume multipart completion supports it.
# https://developers.cloudflare.com/r2/api/s3/extensions/#conditional-operations-in-putobject
MAX_PUT_SIZE = 5 * 1024**3 - 5 * 1024**2
# Published objects. A new pack needs a new key; these must not be replaced.
FROZEN_R2_KEYS = (
    "envs/qwen8b/cu128-py312-torch280-ada.tar.gz",
    "envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz",
    "envs/qwen8b-vllm/cu130-py312-vllm0310-ada.tar.gz",
    "envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz",
    "envs/videojev/cu128-py312-torch280-ada.tar.gz",
    "envs/clip/cu128-py312-torch280-ada.tar.gz",
)
MAX_BUILD_RECORD_SIZE = 64 * 1024
REQUIREMENTS_DIRS = {
    "videojev": "videojev_probe",
    "clip": "clip",
    "siglip2": "siglip2",
    "d3-nano": "d3_nano",
    "qwen8b": "qwen8b",
    "qwen8b-vllm": "qwen8b_vllm",
}
FROZEN_R2_KEYS = (
    "envs/qwen8b/cu128-py312-torch280-ada.tar.gz",
    "envs/qwen8b-vllm/cu128-py312-vllm0310-ada.tar.gz",
    "envs/qwen8b-vllm/cu130-py312-vllm0310-ada.tar.gz",
    "envs/qwen8b-vllm/cu130-py312-vllm0310-ada2.tar.gz",
)
_sha256 = env_source._sha256


def _refuse_frozen_key(r2_key: str) -> None:
    if r2_key in FROZEN_R2_KEYS:
        sys.exit(f"refusing to upload over frozen env key {r2_key}")


def _log(*a):
    print(f"[build_env {time.strftime('%H:%M:%S')}]", *a, flush=True, file=sys.stderr)


def _run(cmd, **kw):
    # Package installers must not inherit storage credentials from the builder.
    child_env = env_source.subprocess_env()
    for key in (
        "PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "LD_LIBRARY_PATH",
        "CC",
        "CXX",
        "MAX_JOBS",
        "HF_HUB_DISABLE_XET",
    ):
        if key in os.environ:
            child_env[key] = os.environ[key]
    child_env.update(kw.pop("env", {}))
    return subprocess.run(
        cmd,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=kw.pop("stdout", sys.stderr),
        env=child_env,
        **kw,
    )


def _ensure_miniconda(home, installer=None, installer_sha256=None):
    home = Path(home).expanduser().resolve()
    conda = home / "bin" / "conda"
    if conda.is_file():
        return conda
    if (
        installer is None
        or not isinstance(installer_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", installer_sha256)
    ):
        raise ValueError(
            "install trusted Miniforge first, or supply a local installer and SHA256"
        )
    installer = Path(installer).resolve()
    if _sha256(installer) != installer_sha256:
        raise ValueError("Miniforge installer sha256 mismatch")
    _run(["bash", str(installer), "-b", "-p", str(home)])
    if not conda.is_file():
        raise RuntimeError("Miniforge did not install conda")
    return conda


def _prefix_python(prefix):
    return str(Path(prefix) / "bin/python")


def _verify_code(flash_attn, compat=None, require_cuda=True, modules=()):
    compat = compat or {"python": "3.12", "torch": "2.8.0", "cuda": "12.8"}
    expected_torch = compat["torch"] + "+cu" + compat["cuda"].replace(".", "")
    lines = [
        "import sys, torch, flash_attn, causal_conv1d"
        if flash_attn
        else "import sys, torch",
        "import os, importlib, json",
        "from importlib.metadata import version",
        "assert sys.flags.isolated and sys.flags.no_user_site, "
        "'isolated Python required'",
        f"assert '.'.join(map(str, sys.version_info[:2])) == {compat['python']!r}, "
        "'unexpected Python'",
        f"assert torch.__version__ == {expected_torch!r}, 'unexpected torch'",
        f"assert torch.version.cuda == {compat['cuda']!r}, 'unexpected CUDA runtime'",
    ]
    if require_cuda:
        lines.append("assert torch.cuda.is_available(), 'CUDA unavailable'")
    lines.extend(
        [
            "for name in "
            f"{list(dict.fromkeys(('torch', 'typing_extensions', *modules)))!r}:",
            "    mod = importlib.import_module(name)",
            "    file = os.path.realpath(getattr(mod, '__file__', '') or '')",
            "    assert file.startswith(os.path.realpath(sys.prefix) + os.sep), "
            "'import outside prefix: ' + name",
            "facts = {",
            "    'python': '.'.join(map(str, sys.version_info[:2])),",
            "    'torch': torch.__version__, 'cuda': torch.version.cuda,",
            "    'compute_capabilities': [], 'extensions': {},",
            "}",
        ]
    )
    if require_cuda:
        lines.append(
            "facts['compute_capabilities'] = sorted({"
            "'.'.join(map(str, torch.cuda.get_device_capability(i))) "
            "for i in range(torch.cuda.device_count())})"
        )
    if flash_attn:
        lines.extend(
            [
                "assert flash_attn.__version__ == '2.8.3.post2', "
                "'unexpected flash-attn'",
                "assert version('causal-conv1d') == '1.7.0', "
                "'unexpected causal-conv1d'",
                "facts['extensions'] = {'flash-attn': flash_attn.__version__, "
                "'causal-conv1d': version('causal-conv1d')}",
            ]
        )
    if "vllm" in compat:
        lines.extend(
            [
                f"assert version('vllm') == {compat['vllm']!r}, 'unexpected vllm'",
                "facts['extensions']['vllm'] = version('vllm')",
            ]
        )
    # Imports (notably serving libraries) may print informational messages.
    # Keep those on stderr; stdout is a strict, single JSON facts channel.
    return (
        "import contextlib, sys\n"
        "with contextlib.redirect_stdout(sys.stderr):\n"
        + "\n".join("    " + line for line in lines)
        + "\nprint(json.dumps(facts))"
    )


def _probe_prefix(prefix, flash_attn, compat=None, require_cuda=True, modules=()):
    guard = (
        "import os, sys\n"
        "sys.dont_write_bytecode = True\n"
        f"assert os.path.realpath(sys.prefix) == {str(Path(prefix).resolve())!r}, "
        "'interpreter prefix mismatch'\n"
    )
    result = _run(
        [
            _prefix_python(prefix),
            "-I",
            "-c",
            guard + _verify_code(flash_attn, compat, require_cuda, modules),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def _build_prefix(
    conda,
    prefix,
    requirements,
    python_ver,
    arch,
    flash_attn=True,
    install_default_torch=True,
    require_cuda=True,
    modules=(),
    compat=None,
):
    """Use argv throughout: paths containing shell syntax remain literal paths.

    ``flash_attn`` gates the slow compile-once step: when False, flash-attn and
    causal-conv1d are NOT installed (and the readiness check does not import
    flash_attn). Models whose runtime never touches flash-attn (e.g. a CLIP/SigLIP
    ViT) set ``env.build.flash_attn: false`` to skip minutes of nvcc compile.
    Every pip/verify step uses the new prefix's ``bin/python -I``, never PATH
    python or ``conda activate``.
    """
    _run(
        [
            str(conda),
            "create",
            "-y",
            "-p",
            str(prefix),
            "-c",
            "conda-forge",
            "--override-channels",
            f"python={python_ver}",
        ]
    )
    python = _prefix_python(prefix)
    pip = [python, "-I", "-m", "pip", "install", "--no-cache-dir"]
    _run([*pip, "--upgrade", "pip", "wheel", "packaging", "ninja"])
    if install_default_torch:
        _run(
            [
                *pip,
                "torch==2.8.0",
                "torchvision==0.23.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ]
        )
    _run([*pip, "-r", str(requirements)])
    if flash_attn:
        _run(
            [
                *pip,
                "--no-build-isolation",
                "flash-attn==2.8.3.post2",
                "causal-conv1d==1.7.0",
            ],
            env={
                "TORCH_CUDA_ARCH_LIST": arch,
                "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
                "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
                "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
            },
        )
    return _probe_prefix(prefix, flash_attn, compat, require_cuda, modules)


def _pack(conda, prefix, out):
    base_python = str(conda.parent / "python")
    _run(
        [
            base_python,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "conda-pack==0.9.2",
        ]
    )
    _run([base_python, "-I", "-m", "conda_pack.cli", "-p", str(prefix), "-o", str(out)])


def _requirements(name, override=None):
    if override is not None:
        path = Path(override)
    else:
        directory = REQUIREMENTS_DIRS.get(name)
        if directory is None:
            raise ValueError("model has no default recipe; specify --requirements")
        path = Path(__file__).parent / directory / "requirements.txt"
    if not path.is_file():
        raise ValueError("requirements file does not exist")
    return path.resolve()


def _flash_attn(env):
    build = env.get("build", {})
    if not isinstance(build, dict) or type(build.get("flash_attn", True)) is not bool:
        raise ValueError("env.build.flash_attn must be a boolean")
    return build.get("flash_attn", True)


def _build_options(env):
    build = env.get("build", {})
    flash = _flash_attn(env)
    for key in ("install_default_torch", "require_cuda"):
        if type(build.get(key, True)) is not bool:
            raise ValueError(f"env.build.{key} must be a boolean")
    modules = []
    for key in ("verify_imports", "verify_modules"):
        values = build.get(key, [])
        if not isinstance(values, list) or not all(
            isinstance(name, str)
            and all(part.isidentifier() for part in name.split("."))
            for name in values
        ):
            raise ValueError(f"env.build.{key} must contain module names")
        modules.extend(values)
    return {
        "install_default_torch": build.get("install_default_torch", True),
        "require_cuda": build.get("require_cuda", True) or flash,
        "verify_modules": list(dict.fromkeys(modules)),
    }


def _metadata(env, name, repo_rev, recipe_sha, *, legacy=False):
    options = _build_options(env)
    defaults = {
        "install_default_torch": True,
        "require_cuda": True,
        "verify_modules": [],
    }
    if legacy and options != defaults:
        raise ValueError("legacy build records cannot describe extended builds")
    metadata = {
        "schema_version": 1 if legacy else 2,
        "name": name,
        "for_model": repo_rev,
        "format": "conda-pack",
        "unpack_to": env["unpack_to"],
        "activate_rel": env["activate_rel"],
        "compat": env["compat"],
        "flash_attn": _flash_attn(env),
        "requirements_sha256": recipe_sha,
    }
    if not legacy:
        metadata["build_options"] = options
        metadata["gpu_validation"] = (
            "measured" if options["require_cuda"] else "deferred"
        )
    return metadata


def _validate_metadata(record, env, name, repo_rev, recipe_sha, extra_keys):
    registry.validate_env(env)
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int:
        raise ValueError("invalid build record")
    expected = _metadata(
        env, name, repo_rev, recipe_sha, legacy=record["schema_version"] == 1
    )
    if (
        set(record) != set(expected) | {"builder_sha256", "runtime"} | set(extra_keys)
        or type(record.get("flash_attn")) is not bool
        or any(record.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("build record does not match model, recipe or compatibility")
    if "build_options" in record:
        for key in ("install_default_torch", "require_cuda"):
            if type(record["build_options"].get(key)) is not bool:
                raise ValueError("invalid build record options")
    builder = record.get("builder_sha256")
    if not isinstance(builder, str) or not re.fullmatch(r"[0-9a-f]{64}", builder):
        raise ValueError("missing builder source identity")
    facts = record.get("runtime")
    compat = env["compat"]
    extensions = (
        {"flash-attn": "2.8.3.post2", "causal-conv1d": "1.7.0"}
        if _flash_attn(env)
        else {}
    )
    if "vllm" in compat:
        extensions["vllm"] = compat["vllm"]
    versions = {
        "python": compat["python"],
        "torch": compat["torch"] + "+cu" + compat["cuda"].replace(".", ""),
        "cuda": compat["cuda"],
        "extensions": extensions,
    }
    if (
        not isinstance(facts, dict)
        or set(facts) != set(versions) | {"compute_capabilities"}
        or any(facts.get(key) != value for key, value in versions.items())
    ):
        raise ValueError("recorded runtime does not match compatibility")
    caps = facts.get("compute_capabilities")
    if _build_options(env)["require_cuda"]:
        if (
            not isinstance(caps, list)
            or not caps
            or any(cap not in registry.GPU_ARCHES[compat["gpu_family"]] for cap in caps)
            or caps != sorted(set(caps))
        ):
            raise ValueError("recorded GPU does not match compatibility")
    elif caps != []:
        raise ValueError("CPU build must leave GPU validation deferred")


def _record_path(tarball):
    return Path(str(tarball) + ".build.json")


def _check_pack(tarball):
    """Read headers/critical entries only; never extract or execute a pack.

    Restore's full safe extraction remains authoritative. This preflight catches
    incomplete or non-conda archives before they can be proposed as registry pins.
    """
    with tarfile.open(tarball, "r:gz") as archive:
        members = {}
        for member in archive:
            if member.isdir() and member.name in {".", "./"}:
                continue
            name = member.name.rstrip("/") if member.isdir() else member.name
            path = registry.relative_path(name)
            if name in members or path.parts[0] == env_source.MARKER:
                raise ValueError("duplicate packed archive path or completion marker")
            if not (
                member.isfile() or member.isdir() or member.issym() or member.islnk()
            ):
                raise ValueError("packed archive contains a special file")
            members[name] = member
        for name, member in members.items():
            for parent in PurePosixPath(name).parents:
                if str(parent) in members and not members[str(parent)].isdir():
                    raise ValueError("packed archive entry has a non-directory parent")
            if member.issym() or member.islnk():
                target = member.linkname
                if (
                    not target
                    or target.startswith("/")
                    or any(c in target for c in "\\:\x00\r\n")
                ):
                    raise ValueError("unsafe packed archive link")
                if member.islnk():
                    registry.relative_path(target)
                    if target not in members or not members[target].isfile():
                        raise ValueError("packed hardlink must target a regular member")
        env_source._validate_archive_links(members)
        for critical in ("bin/python", "bin/activate", "bin/conda-unpack"):
            # Match restore semantics: resolve each symlink before processing '..'.
            pending, resolved, hops = deque(critical.split("/")), [], 0
            while pending:
                part = pending.popleft()
                if part in {"", "."}:
                    continue
                if part == "..":
                    if not resolved:
                        raise ValueError("packed archive link escapes the environment")
                    resolved.pop()
                    continue
                member = members.get("/".join([*resolved, part]))
                if member is not None and member.issym():
                    hops += 1
                    if hops > 40:
                        raise ValueError("packed archive link cycle")
                    pending.extendleft(reversed(member.linkname.split("/")))
                else:
                    resolved.append(part)
            member = members.get("/".join(resolved))
            if member is not None and member.islnk():
                member = members[member.linkname]
            if member is None or not member.isfile():
                raise ValueError(f"missing packed archive file: {critical}")
            if critical == "bin/python" and not member.mode & 0o111:
                raise ValueError("packed Python is not executable")


def _validate_record(record, env, name, tarball, repo_rev, recipe_sha):
    """Bind locally trusted build evidence to bytes, model and selected recipe.

    Records are not signatures or independent ABI acceptance. Never reconstruct
    missing build facts from the upload host or the requested registry entry.
    """
    if not isinstance(record, dict):
        raise ValueError("invalid build record")
    extra = {"sha256", "size"}
    if "prefix_build" in record:
        if record.get("schema_version") != 2:
            raise ValueError("staged packs require a version 2 record")
        extra.add("prefix_build")
        _validate_prefix_record(record["prefix_build"], env, name, repo_rev, recipe_sha)
    _validate_metadata(record, env, name, repo_rev, recipe_sha, extra)
    tarball = Path(tarball)
    if tarball.is_symlink() or not tarball.is_file():
        raise ValueError("pack must be a regular file, not a symlink")
    if (
        type(record.get("size")) is not int
        or record["size"] != tarball.stat().st_size
        or record.get("sha256") != _sha256(tarball)
    ):
        raise ValueError("pack does not match build record digest/size")
    _check_pack(tarball)


def _new_record(
    env, name, tarball, repo_rev, recipe_sha, runtime, *, prefix_build=None
):
    record = {
        **_metadata(env, name, repo_rev, recipe_sha),
        "builder_sha256": _sha256(Path(__file__)),
        "runtime": runtime,
        "sha256": _sha256(tarball),
        "size": tarball.stat().st_size,
    }
    if prefix_build is not None:
        record["prefix_build"] = prefix_build
    _validate_record(record, env, name, tarball, repo_rev, recipe_sha)
    with _record_path(tarball).open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2)
    return record


def _read_record(tarball):
    path = _record_path(tarball)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular build record: {path}; rebuild the pack")
    with path.open("rb") as source:
        raw = source.read(MAX_BUILD_RECORD_SIZE + 1)
    if len(raw) > MAX_BUILD_RECORD_SIZE:
        raise ValueError("build record exceeds 64 KiB")
    return json.loads(raw)


@contextmanager
def _upload_snapshot(tarball):
    """Freeze bounded local bytes before any slow credential/client operation."""
    tarball = Path(tarball)
    if tarball.is_symlink() or not tarball.is_file():
        raise ValueError("pack must be a regular file")
    with tempfile.TemporaryDirectory(prefix="env-upload-") as directory:
        snapshot = Path(directory) / "pack.tar.gz"
        size = 0
        with tarball.open("rb") as source, snapshot.open("xb") as output:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                size += len(block)
                if size > MAX_PUT_SIZE:
                    raise ValueError(
                        "create-only upload supports packs below 5 GiB; "
                        "no multipart fallback"
                    )
                output.write(block)
        if not size:
            raise ValueError("create-only upload requires a nonempty pack")
        yield snapshot


def _upload(env_spec, name, tarball, compat, repo_rev, *, record, recipe_sha):
    """Create content-addressed objects conditionally; never delete after a timeout.

    A failed/uncertain second write can leave an unreferenced tarball. Preserve it
    for author inspection rather than risking deletion of a committed artifact.
    """
    registry.validate_env(env_spec)
    _refuse_frozen_key(env_spec["r2_key"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("invalid model name")
    if compat != env_spec["compat"]:
        raise ValueError("build compatibility differs from registry")
    with _upload_snapshot(tarball) as tarball:
        _validate_record(record, env_spec, name, tarball, repo_rev, recipe_sha)
        sha, size = record["sha256"], record["size"]
        key = f"envs/{name}/{sha}.tar.gz"
        manifest_key = f"envs/{name}/{sha}.json"
        manifest = {
            "name": name,
            "r2_key": key,
            "sha256": sha,
            "size": size,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "unpack_to": env_spec["unpack_to"],
            "activate_rel": env_spec["activate_rel"],
            "compat": compat,
            "built_by": "goodvision-harness build_env",
            "for_model": repo_rev,
            "format": "conda-pack",
            "build": record,
        }
        client = model_source.r2_client()
        if client is None:
            raise RuntimeError("R2 credentials not configured")
        bucket = os.environ["R2_BUCKET"]
        try:
            with tarball.open("rb") as body:
                client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=body,
                    ContentLength=size,
                    IfNoneMatch="*",
                )
            client.put_object(
                Bucket=bucket,
                Key=manifest_key,
                IfNoneMatch="*",
                Body=json.dumps(manifest, indent=2).encode(),
            )
        except Exception:
            raise RuntimeError(
                f"create-only upload failed or is uncertain; inspect {key} "
                f"and {manifest_key}; "
                "existing objects were not deleted or overwritten"
            ) from None
        return {**manifest, "manifest_key": manifest_key}


def _assert_prefix_files(prefix: Path, *, verify_record_hashes=True) -> dict:
    """Fail closed on conda/pip file holes before conda-pack.

    Checks: conda-meta listed files exist; pip RECORD hashes match on-disk
    .py/.so (skip __pycache__ and foreign-python .pyc). After conda-unpack,
    intentional prefix relocation invalidates some original RECORD hashes;
    that check is explicitly skipped, while missing-file and import checks stay.
    Neither path establishes GPU ABI acceptance.
    """
    import base64
    import csv
    from collections import defaultdict

    prefix = Path(prefix).resolve()
    conda_meta = prefix / "conda-meta"
    missing_conda = defaultdict(list)
    relocated = {}
    if conda_meta.is_dir():
        for spec in sorted(conda_meta.glob("*.json")):
            try:
                info = json.loads(spec.read_text())
            except (OSError, ValueError) as error:
                raise ValueError("invalid conda package metadata") from error
            if not isinstance(info, dict) or not isinstance(info.get("files"), list):
                raise ValueError("invalid conda package file list")
            for entry in info.get("paths_data", {}).get("paths", []):
                if not isinstance(entry, dict):
                    raise ValueError("invalid conda path metadata")
                rel = entry.get("_path")
                registry.relative_path(rel)
                if (
                    entry.get("prefix_placeholder")
                    and entry.get("file_mode") in {"text", "binary"}
                    and all(
                        isinstance(entry.get(key), str)
                        and re.fullmatch(r"[0-9a-f]{64}", entry[key])
                        for key in ("sha256", "sha256_in_prefix")
                    )
                ):
                    relocated[(prefix / rel).resolve()] = (
                        entry["sha256"],
                        entry["sha256_in_prefix"],
                    )
            pkg = f"{info.get('name')}=={info.get('version')}"
            for rel in info.get("files") or []:
                registry.relative_path(rel)
                if not rel or rel.endswith("/") or "__pycache__" in rel.split("/"):
                    continue
                target = prefix / rel
                # Directories and prefix-relative symlinks (terminfo, python3.1)
                # are valid conda entries; is_file() is false for those.
                if target.exists() or target.is_symlink():
                    continue
                missing_conda[pkg].append(rel)
    if missing_conda:
        preview = {k: v[:8] for k, v in list(missing_conda.items())[:6]}
        raise RuntimeError(
            "conda-meta lists files missing from the prefix (pip likely "
            f"clobbered a conda package). conda-pack would drop them if "
            f"ignore-missing-files were set: {preview}"
        )

    site = None
    for cand in sorted(prefix.glob("lib/python*/site-packages")):
        if cand.is_dir():
            site = cand
            break
    record_mismatch = []
    record_missing = []
    checked = 0
    checked_relocations = 0
    if site is not None:
        for dist in sorted(site.glob("*.dist-info")):
            rec = dist / "RECORD"
            if not rec.is_file():
                continue
            with rec.open(newline="") as fh:
                for row in csv.reader(fh):
                    if not row:
                        continue
                    rel, digest = row[0], row[1] if len(row) > 1 else ""
                    if not rel or rel.endswith("/") or "__pycache__" in rel.split("/"):
                        continue
                    if rel.endswith(".pyc"):
                        continue
                    path = site / rel
                    if not path.resolve().is_relative_to(prefix):
                        raise ValueError("pip RECORD path escapes the prefix")
                    if not path.is_file():
                        if rel.endswith((".py", ".so")):
                            record_missing.append(f"{dist.name}:{rel}")
                        continue
                    if (
                        not verify_record_hashes
                        or not digest.startswith("sha256=")
                        or not rel.endswith((".py", ".so"))
                    ):
                        continue
                    checked += 1
                    actual_sha = _sha256(path)
                    got = (
                        base64.urlsafe_b64encode(bytes.fromhex(actual_sha))
                        .decode()
                        .rstrip("=")
                    )
                    if f"sha256={got}" != digest:
                        relocation = relocated.get(path.resolve())
                        if (
                            relocation
                            and relocation[1] == actual_sha
                            and digest
                            == (
                                "sha256="
                                + base64.urlsafe_b64encode(bytes.fromhex(relocation[0]))
                                .decode()
                                .rstrip("=")
                            )
                        ):
                            checked_relocations += 1
                        else:
                            record_mismatch.append(f"{dist.name}:{rel}")
    if record_missing:
        raise RuntimeError(
            "pip RECORD lists .py/.so files missing from the prefix "
            f"(missing={record_missing[:12]})"
        )
    if record_mismatch:
        raise RuntimeError(
            "pip RECORD hash mismatch in build prefix; inspect the package "
            "and any conda relocation before rebuilding: " + str(record_mismatch[:12])
        )
    return {
        "conda_missing": 0,
        "record_py_so_checked": checked,
        "record_missing": 0,
        "record_mismatch": len(record_mismatch),
        "record_hashes_verified": verify_record_hashes,
        "record_relocations_verified": checked_relocations,
    }


def _refuse_frozen_key(r2_key):
    if r2_key in FROZEN_R2_KEYS:
        sys.exit(
            f"refusing to upload over frozen env key {r2_key}; "
            "select a reviewed successor"
        )


def _verify_packed_build(out, env, runtime):
    """Check a fresh safe restore using only the packed isolated interpreter."""
    options = _build_options(env)
    _check_pack(out)
    with tempfile.TemporaryDirectory(
        prefix="verify-conda-pack-", dir=out.parent
    ) as temp:
        restored = Path(temp) / "env"
        env_source._safe_extract(out, restored)
        env_source._run_unpack(restored)
        _assert_prefix_files(restored, verify_record_hashes=False)
        measured = _probe_prefix(
            restored,
            _flash_attn(env),
            env["compat"],
            options["require_cuda"],
            options["verify_modules"],
        )
        if measured != runtime:
            raise ValueError("restored runtime differs from build prefix")


def _prefix_record_path(out):
    return Path(str(out) + ".prefix.json")


def _prefix_fingerprint(prefix):
    """Bind source files, modes and literal internal links without following dirs."""
    prefix = Path(prefix)
    if prefix.is_symlink() or not prefix.is_dir():
        raise ValueError("prefix fingerprint requires a regular directory")
    digest = hashlib.sha256()
    inodes = {}
    resolved_prefix = prefix.resolve()

    def enumeration_failed(error):
        raise ValueError("cannot read the complete build prefix") from error

    for directory, dirs, files in os.walk(
        prefix, followlinks=False, onerror=enumeration_failed
    ):
        for name in sorted([*dirs, *files]):
            path = Path(directory) / name
            metadata = path.lstat()
            row = {
                "path": path.relative_to(prefix).as_posix(),
                "mode": stat.S_IMODE(metadata.st_mode),
            }
            if stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(path)
                if os.path.isabs(target) or not path.resolve().is_relative_to(
                    resolved_prefix
                ):
                    raise ValueError("unsafe link in build prefix")
                row.update(kind="link", target=target)
            elif stat.S_ISDIR(metadata.st_mode):
                row["kind"] = "directory"
            elif stat.S_ISREG(metadata.st_mode):
                row.update(kind="file", sha256=_sha256(path))
                inode = (metadata.st_dev, metadata.st_ino)
                if inode in inodes:
                    row["hardlink_to"] = inodes[inode]
                else:
                    inodes[inode] = row["path"]
            else:
                raise ValueError("special file in build prefix")
            digest.update(json.dumps(row, sort_keys=True).encode() + b"\n")
        dirs.sort()
    return digest.hexdigest()


def _validate_prefix_record(record, env, name, repo_rev, recipe_sha):
    _validate_metadata(record, env, name, repo_rev, recipe_sha, {"prefix_sha256"})
    fingerprint = record.get("prefix_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", fingerprint
    ):
        raise ValueError("missing source prefix fingerprint")


def _read_prefix_record(out, env, name, repo_rev, recipe_sha):
    path = _prefix_record_path(out)
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "pack-only requires the trusted prefix record from --skip-pack"
        )
    with path.open("rb") as source:
        raw = source.read(MAX_BUILD_RECORD_SIZE + 1)
    if len(raw) > MAX_BUILD_RECORD_SIZE:
        raise ValueError("prefix record exceeds 64 KiB")
    record = json.loads(raw)
    _validate_prefix_record(record, env, name, repo_rev, recipe_sha)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build a new immutable conda-pack environment."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--conda-home", default=str(Path.home() / "miniconda3"))
    ap.add_argument("--miniforge-installer", type=Path)
    ap.add_argument("--miniforge-sha256")
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="tarball path (must not exist for build; must exist for --upload-only)",
    )
    ap.add_argument(
        "--requirements",
        type=Path,
        help="override the selected model's recipe (required again for upload-only)",
    )
    ap.add_argument(
        "--skip-upload",
        action="store_true",
        help="build + pack only; keep the tarball at --out",
    )
    ap.add_argument(
        "--upload-only",
        action="store_true",
        help="upload --out with its matching .build.json record; do not rebuild",
    )
    ap.add_argument(
        "--skip-pack",
        action="store_true",
        help="build and record a prefix; do not pack or upload",
    )
    ap.add_argument(
        "--pack-only",
        action="store_true",
        help="pack a prefix recorded by --skip-pack; verify it again",
    )
    args = ap.parse_args(argv)
    if sum((args.skip_pack, args.pack_only, args.upload_only)) > 1:
        ap.error("--skip-pack, --pack-only and --upload-only are mutually exclusive")
    if args.skip_upload and args.upload_only:
        ap.error("--skip-upload and --upload-only are mutually exclusive")
    env_spec = registry.env_of(args.model)
    if env_spec is None:
        ap.error("model has no env block")
    compat = env_spec["compat"]
    options = _build_options(env_spec)
    if compat["python"] != "3.12":
        ap.error("this builder supports only Python 3.12")
    if options["install_default_torch"] and (compat["torch"], compat["cuda"]) != (
        "2.8.0",
        "12.8",
    ):
        ap.error("default torch installer requires torch 2.8.0 / CUDA 12.8")
    if not args.skip_upload and not args.skip_pack:
        _refuse_frozen_key(env_spec["r2_key"])
    repo, rev = registry.weights_of(args.model)
    repo_rev = f"{repo}@{rev}"
    requirements = _requirements(args.model, args.requirements)
    recipe_sha = _sha256(requirements)
    prefix = Path(env_spec["unpack_to"])
    out = args.out.absolute()
    arch = ";".join(registry.GPU_ARCHES[compat["gpu_family"]])
    flash_attn = _flash_attn(env_spec)
    if args.upload_only:
        if not out.is_file():
            ap.error(
                f"--upload-only: no tarball at {out}; build it first "
                "(e.g. with --skip-upload)"
            )
        _log(f"upload-only {args.model}: reusing existing {out}")
        record = _read_record(out)
    else:
        if options["require_cuda"]:
            env_source.check_compatibility(env_spec)
        elif platform.system() != "Linux" or platform.machine() != "x86_64":
            ap.error("CPU wheel builds require Linux x86_64")
        if any(p.exists() or p.is_symlink() for p in (out, _record_path(out))):
            ap.error("output already exists; choose a new path")
        if out.resolve().is_relative_to(prefix.resolve()):
            ap.error("output must be outside the environment prefix")
        with env_source.prefix_lock(prefix), env_source.interruptible():
            stage_path = _prefix_record_path(out)
            previous = None
            if args.pack_only:
                previous = _read_prefix_record(
                    out, env_spec, args.model, repo_rev, recipe_sha
                )
                if prefix.is_symlink() or not prefix.is_dir():
                    ap.error("pack-only requires the recorded existing prefix")
                if _prefix_fingerprint(prefix) != previous["prefix_sha256"]:
                    raise ValueError("prefix contents changed since --skip-pack")
                runtime = _probe_prefix(
                    prefix,
                    flash_attn,
                    compat,
                    options["require_cuda"],
                    options["verify_modules"],
                )
                if runtime != previous["runtime"]:
                    raise ValueError("prefix runtime changed since --skip-pack")
            else:
                if prefix.exists() or prefix.is_symlink():
                    ap.error(
                        "preserving existing environment; "
                        "inspect and move it aside first"
                    )
                if stage_path.exists() or stage_path.is_symlink():
                    ap.error("prefix record already exists; choose a new path")
            conda = _ensure_miniconda(
                args.conda_home, args.miniforge_installer, args.miniforge_sha256
            )
            if not args.pack_only:
                runtime = _build_prefix(
                    conda,
                    prefix,
                    requirements,
                    compat["python"],
                    arch,
                    flash_attn,
                    options["install_default_torch"],
                    options["require_cuda"],
                    options["verify_modules"],
                    compat,
                )
            _assert_prefix_files(prefix)
            if _sha256(requirements) != recipe_sha:
                raise ValueError("requirements changed during build; refusing a record")
            if args.skip_pack:
                record = {
                    **_metadata(env_spec, args.model, repo_rev, recipe_sha),
                    "builder_sha256": _sha256(Path(__file__)),
                    "runtime": runtime,
                    "prefix_sha256": _prefix_fingerprint(prefix),
                }
                _validate_prefix_record(
                    record, env_spec, args.model, repo_rev, recipe_sha
                )
                with stage_path.open("x", encoding="utf-8") as output:
                    json.dump(record, output, indent=2)
                print(
                    json.dumps(
                        {
                            "model": args.model,
                            "prefix": str(prefix),
                            "prefix_record": str(stage_path),
                            "packed": False,
                            "uploaded": False,
                        }
                    )
                )
                return
            if (
                previous is not None
                and _prefix_fingerprint(prefix) != previous["prefix_sha256"]
            ):
                raise ValueError("prefix contents changed during verification")
            _pack(conda, prefix, out)
            if (
                previous is not None
                and _prefix_fingerprint(prefix) != previous["prefix_sha256"]
            ):
                raise ValueError("prefix contents changed during packing")
            _verify_packed_build(out, env_spec, runtime)
            if _sha256(requirements) != recipe_sha:
                raise ValueError("requirements changed during build; refusing a record")
            record = _new_record(
                env_spec,
                args.model,
                out,
                repo_rev,
                recipe_sha,
                runtime,
                prefix_build=previous,
            )
    sha = _sha256(out)
    result = {
        "model": args.model,
        "tarball": str(out),
        "build_record": str(_record_path(out)),
        "sha256": sha,
        "size": out.stat().st_size,
        "arch": arch,
        "flash_attn": flash_attn,
        "uploaded": False,
    }
    if not args.skip_upload:
        result.update(
            _upload(
                env_spec,
                args.model,
                out,
                compat,
                repo_rev,
                record=record,
                recipe_sha=recipe_sha,
            )
        )
        result["uploaded"] = True
    print(json.dumps(result))


if __name__ == "__main__":
    main()
