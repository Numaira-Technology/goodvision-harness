"""Build once on a compatible GPU and publish NEW immutable environment objects.

No weights are built or uploaded. Existing prefixes, packs and R2 objects are
never replaced. The output is a proposed registry pin; review it before use.
A trusted installed Miniforge, or a local installer plus its independently
verified SHA256, is required. Nothing is fetched from a floating installer URL.
"""

import argparse
import json
import os
import re
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
MAX_BUILD_RECORD_SIZE = 64 * 1024
REQUIREMENTS_DIRS = {
    "videojev": "videojev_probe",
    "clip": "clip",
    "siglip2": "siglip2",
    "d3-nano": "d3_nano",
    "qwen8b": "qwen8b",
}
_sha256 = env_source._sha256


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


def _verify_code(flash_attn):
    lines = [
        "import sys, torch",
        "assert sys.version_info[:2] == (3, 12), 'unexpected Python'",
        "assert torch.__version__ == '2.8.0+cu128', 'unexpected torch'",
        "assert torch.version.cuda == '12.8', 'unexpected CUDA runtime'",
        "assert torch.cuda.is_available(), 'CUDA unavailable'",
    ]
    if flash_attn:
        lines[0] = "import sys, torch, flash_attn, causal_conv1d"
        lines.insert(
            4,
            "assert flash_attn.__version__ == '2.8.3.post2', 'unexpected flash-attn'",
        )
        lines.extend(
            [
                "from importlib.metadata import version",
                "assert version('causal-conv1d') == '1.7.0', "
                "'unexpected causal-conv1d'",
            ]
        )
    lines.extend(
        [
            "import json",
            "facts = {",
            "    'python': '.'.join(map(str, sys.version_info[:2])),",
            "    'torch': torch.__version__, 'cuda': torch.version.cuda,",
            "    'compute_capabilities': sorted({",
            "        '.'.join(map(str, torch.cuda.get_device_capability(i)))",
            "        for i in range(torch.cuda.device_count())}),",
            "    'extensions': {},",
            "}",
        ]
    )
    if flash_attn:
        lines.append(
            "facts['extensions'] = {'flash-attn': flash_attn.__version__, "
            "'causal-conv1d': version('causal-conv1d')}"
        )
    # Dependencies may print import diagnostics. Keep stdout as one JSON record.
    return (
        "import contextlib, sys\n"
        "with contextlib.redirect_stdout(sys.stderr):\n"
        + "\n".join("    " + line for line in lines)
        + "\nprint(json.dumps(facts))"
    )


def _build_prefix(conda, prefix, requirements, python_ver, arch, flash_attn=True):
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
    _run([*pip, "--upgrade", "pip", "setuptools", "wheel", "packaging", "ninja"])
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
    result = _run(
        [python, "-I", "-c", _verify_code(flash_attn)],
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


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
    registry.validate_env(env)
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int:
        raise ValueError("invalid build record")
    expected = {
        "schema_version": 1,
        "name": name,
        "for_model": repo_rev,
        "format": "conda-pack",
        "unpack_to": env["unpack_to"],
        "activate_rel": env["activate_rel"],
        "compat": env["compat"],
        "flash_attn": _flash_attn(env),
        "requirements_sha256": recipe_sha,
    }
    if (
        set(record) != set(expected) | {"builder_sha256", "runtime", "sha256", "size"}
        or type(record.get("flash_attn")) is not bool
        or any(record.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("build record does not match model, recipe or compatibility")
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
    if (
        not isinstance(caps, list)
        or not caps
        or any(cap not in registry.GPU_ARCHES[compat["gpu_family"]] for cap in caps)
        or caps != sorted(set(caps))
    ):
        raise ValueError("recorded GPU does not match compatibility")
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


def _new_record(env, name, tarball, repo_rev, recipe_sha, runtime):
    record = {
        "schema_version": 1,
        "name": name,
        "for_model": repo_rev,
        "format": "conda-pack",
        "unpack_to": env["unpack_to"],
        "activate_rel": env["activate_rel"],
        "compat": env["compat"],
        "flash_attn": _flash_attn(env),
        "requirements_sha256": recipe_sha,
        "builder_sha256": _sha256(Path(__file__)),
        "runtime": runtime,
        "sha256": _sha256(tarball),
        "size": tarball.stat().st_size,
    }
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
    args = ap.parse_args(argv)
    if args.skip_upload and args.upload_only:
        ap.error("--skip-upload and --upload-only are mutually exclusive")
    env_spec = registry.env_of(args.model)
    if env_spec is None:
        ap.error("model has no env block")
    compat = env_spec["compat"]
    if any(
        compat[k] != v
        for k, v in {"python": "3.12", "torch": "2.8.0", "cuda": "12.8"}.items()
    ):
        ap.error("this builder supports only Python 3.12 / torch 2.8.0 / CUDA 12.8")
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
        env_source.check_compatibility(env_spec)
        if any(p.exists() or p.is_symlink() for p in (out, _record_path(out))):
            ap.error("output already exists; choose a new path")
        if out.is_relative_to(prefix):
            ap.error("output must be outside the environment prefix")
        with env_source.prefix_lock(prefix), env_source.interruptible():
            if prefix.exists():
                ap.error(
                    "preserving existing environment; inspect and move it aside first"
                )
            conda = _ensure_miniconda(
                args.conda_home, args.miniforge_installer, args.miniforge_sha256
            )
            runtime = _build_prefix(
                conda,
                prefix,
                requirements,
                compat["python"],
                arch,
                flash_attn,
            )
            _pack(conda, prefix, out)
            if _sha256(requirements) != recipe_sha:
                raise ValueError("requirements changed during build; refusing a record")
            record = _new_record(
                env_spec, args.model, out, repo_rev, recipe_sha, runtime
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
