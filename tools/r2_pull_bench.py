"""A/B benchmark of R2 weight-download strategies on one rented Vast box.

Answers "is the remaining ceiling our Python downloader or the network?" by
pulling the same pinned model several ways on the same machine:

    threads:FxS   model_source.pull_from_r2, F files in threads x S streams
    env:NAME      env_source restore of NAME's packed env from models.yaml, timed
                  per step (download, verify, extract, conda-unpack)
    both:NAME     that env restore and the weight pull at the same time, as the
                  qwen8b launchers now do
    procs:PxS     model_source.pull_from_r2, P processes x S streams
    s5cmd:WxC     s5cmd (Go) with W workers x C parts per file, then the same
                  size+sha256 check model_source does (timed separately)

Two halves:

    rent  (laptop)  rents ONE offer you already chose and approved, uploads
                    this file + model_source.py, runs ``box`` over SSH with
                    R2 credentials on stdin, and always destroys the instance.
    box   (GPU box) runs the variants and prints one RESULT line each.

Example (offer id and price cap come from you, never chosen here):

    PYTHONPATH=src:. python tools/r2_pull_bench.py rent --offer 47594089 \\
        --max-dph 0.12 --variants threads:4x16 threads:4x4 procs:4x8 s5cmd:16x16

Needs VAST_API_KEY, SAM_SSH_KEY and R2_* in the repository .env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from uuid import uuid4

R2_KEYS = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
DEFAULT_REPO = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
LABEL_PREFIX = "goodvision-r2bench-"
# Files the box half imports (tools/ package layout, mirrored on the box).
BOX_TOOLS = ("model_source.py", "env_source.py", "registry.py", "models.yaml")
BOX_TOOLS_UB = ("unified_benchmark/__init__.py", "unified_benchmark/locking.py")
IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"  # has python3 + pip
# Pinned s5cmd release; hash from the release's s5cmd_checksums.txt.
S5CMD_URL = (
    "https://github.com/peak/s5cmd/releases/download/v2.3.0/"
    "s5cmd_2.3.0_Linux-64bit.tar.gz"
)
S5CMD_SHA256 = "de0fdbfa3aceae55e069ba81a0fc17b2026567637603734a387b2fca06c299b4"


def parse_variant(text):
    kind, _, shape = text.partition(":")
    if kind in {"env", "both"}:
        if not shape or not all(c.isalnum() or c in "-_" for c in shape):
            raise argparse.ArgumentTypeError(f"bad variant {text!r} (e.g. env:qwen8b)")
        return kind, shape, None
    a, _, b = shape.partition("x")
    if kind not in {"threads", "procs", "s5cmd"} or not (a.isdigit() and b.isdigit()):
        raise argparse.ArgumentTypeError(f"bad variant {text!r} (e.g. threads:4x16)")
    return kind, int(a), int(b)


# --------------------------------------------------------------------- box side


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _s5cmd(workdir):
    binary = Path(workdir) / "s5cmd"
    if binary.is_file():
        return binary
    archive = Path(workdir) / "s5cmd.tar.gz"
    with urllib.request.urlopen(S5CMD_URL, timeout=120) as r, open(archive, "wb") as f:
        shutil.copyfileobj(r, f)
    if _sha256(archive) != S5CMD_SHA256:
        raise RuntimeError("s5cmd download failed its pinned sha256 check")
    with tarfile.open(archive) as tar:
        member = tar.getmember("s5cmd")
        member.name = "s5cmd"
        try:
            tar.extract(member, workdir, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extract(member, workdir)
    binary.chmod(0o755)
    return binary


def _pull_s5cmd(ms, repo, revision, out, workers, parts, workdir):
    """Download with s5cmd, then verify exactly like model_source."""
    env = dict(
        os.environ,
        AWS_ACCESS_KEY_ID=os.environ["R2_ACCESS_KEY_ID"],
        AWS_SECRET_ACCESS_KEY=os.environ["R2_SECRET_ACCESS_KEY"],
        AWS_REGION="auto",
    )
    client = ms.r2_client()
    bucket = os.environ["R2_BUCKET"]
    prefix = f"{ms.R2_PREFIX}/{repo}/{revision}/"
    manifest = json.loads(
        client.get_object(Bucket=bucket, Key=prefix + ms.MANIFEST_NAME)["Body"].read()
    )
    files = ms._manifest_files(manifest)
    t = time.monotonic()
    subprocess.run(
        [
            str(_s5cmd(workdir)),
            "--endpoint-url",
            os.environ["R2_ENDPOINT"],
            "--numworkers",
            str(workers),
            "cp",
            "--concurrency",
            str(parts),
            f"s3://{bucket}/{prefix}*",
            str(out) + "/",
        ],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    download = time.monotonic() - t
    with ProcessPoolExecutor() as pool:
        digests = dict(
            zip(files, pool.map(_sha256, [Path(out) / r for r in files]), strict=True)
        )
    for rel, meta in files.items():
        p = Path(out) / rel
        if p.stat().st_size != meta["size"] or digests[rel] != meta["sha256"]:
            raise ValueError(f"s5cmd result failed verification: {rel}")
    return manifest, download


def box(argv):
    ap = argparse.ArgumentParser(prog="r2_pull_bench box")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument(
        "--env-root",
        help="restore envs under this directory instead of unpack_to (dry runs)",
    )
    ap.add_argument("variants", nargs="+", type=parse_variant)
    args = ap.parse_args(argv)
    os.environ.update(json.loads(sys.stdin.readline()))  # R2_* only, via stdin
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))  # uploaded tools/ package lives next to us
    from tools import model_source as ms

    print(f"cpus={os.cpu_count()} disk={shutil.disk_usage(here).free / 1e9:.0f}GB")
    out = here / "bench-out"
    for kind, a, b in args.variants:
        shutil.rmtree(out, ignore_errors=True)
        for name in ("R2_PULL_FILE_WORKERS", "R2_PULL_STREAMS", "R2_PULL_PROCESSES"):
            os.environ.pop(name, None)
        note = ""
        t = time.monotonic()
        if kind in {"env", "both"}:
            _env_variant(kind, a, ms, args, out, env_root=args.env_root)
            continue
        try:
            if kind == "s5cmd":
                manifest, download = _pull_s5cmd(
                    ms, args.repo, args.revision, out, a, b, here
                )
                note = f" (download {download:.0f}s + verify)"
            else:
                if kind == "procs":
                    os.environ["R2_PULL_PROCESSES"] = str(a)
                else:  # processes are the default now; pin threads explicitly
                    os.environ["R2_PULL_PROCESSES"] = "1"
                    os.environ["R2_PULL_FILE_WORKERS"] = str(a)
                os.environ["R2_PULL_STREAMS"] = str(b)
                manifest = ms.pull_from_r2(args.repo, args.revision, out)
        except Exception as e:  # report and keep going; no exception text (secrets)
            print(f"RESULT {kind}:{a}x{b}: FAILED {type(e).__name__}", flush=True)
            continue
        dt = time.monotonic() - t
        total = sum(m["size"] for m in manifest["files"].values())
        print(
            f"RESULT {kind}:{a}x{b}: {total / 1e9:.2f} GB in {dt:.0f}s = "
            f"{total / 1e6 / dt:.0f} MB/s{note}",
            flush=True,
        )
    shutil.rmtree(out, ignore_errors=True)


def _env_variant(kind, name, ms, args, out, env_root=None):
    """Time a packed-env restore, alone or beside the weight pull."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tools import env_source as es

    env = es.registry.env_of(name)
    if env_root:
        # Dry run off the box: unpack_to is part of the pinned identity, so the
        # relocated prefix must not be compared with the bucket manifest.
        env = {**env, "unpack_to": str(Path(env_root).resolve() / name)}
        es.IDENTITY_FIELDS = tuple(f for f in es.IDENTITY_FIELDS if f != "unpack_to")
    prefix = Path(env["unpack_to"])
    # Timing only: the cheap benchmark box need not match env.compat.gpu_family;
    # download, verify, extract and conda-unpack do not touch the GPU.
    es.check_compatibility = lambda env: None
    steps, lock = {}, threading.Lock()

    def timed(label, fn):
        def wrapper(*a, **kw):
            start = time.monotonic()
            try:
                return fn(*a, **kw)
            finally:
                with lock:
                    steps[label] = steps.get(label, 0) + time.monotonic() - start

        return wrapper

    labels = {
        "_sha256": "verify",
        "_safe_extract": "extract",
        "_run_unpack": "conda-unpack",
    }
    # Keep the originals once, so repeated variants never wrap a wrapper.
    originals = es.__dict__.setdefault(
        "_bench_originals", {attr: getattr(es, attr) for attr in labels}
    )
    for attr, label in labels.items():
        setattr(es, attr, timed(label, originals[attr]))

    class TimedClient:
        def __init__(self, client):
            self._client = client
            self.download_file = timed("download", client.download_file)

        def __getattr__(self, attr):
            return getattr(self._client, attr)

    def restore():
        es.pull_env(env, client=TimedClient(ms.r2_client()))
        return time.monotonic() - start

    def weights():
        for knob in ("R2_PULL_FILE_WORKERS", "R2_PULL_STREAMS", "R2_PULL_PROCESSES"):
            os.environ.pop(knob, None)
        ms.pull_from_r2(args.repo, args.revision, out)
        return time.monotonic() - start

    start = time.monotonic()
    try:
        shutil.rmtree(prefix, ignore_errors=True)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        if kind == "env":
            env_s = restore()
            detail = ""
        else:
            with ThreadPoolExecutor(2) as pool:
                env_f, w_f = pool.submit(restore), pool.submit(weights)
                env_s, w_s = env_f.result(), w_f.result()
            detail = f"; weights done at {w_s:.0f}s"
    except Exception as e:  # no exception text (secrets)
        done = ", ".join(f"{k} {v:.0f}s" for k, v in steps.items()) or "none"
        print(
            f"RESULT {kind}:{name}: FAILED {type(e).__name__} (steps timed: {done})",
            flush=True,
        )
        return
    finally:
        shutil.rmtree(prefix, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)
    total = time.monotonic() - start
    parts = ", ".join(f"{k} {v:.0f}s" for k, v in steps.items())
    print(
        f"RESULT {kind}:{name}: total {total:.0f}s; env done at {env_s:.0f}s "
        f"({env['size'] / 1e9:.2f} GB: {parts}){detail}",
        flush=True,
    )


# ------------------------------------------------------------------ laptop side


def find_approved_offer(vast, args):
    """The approved offer, or (with --machine) a twin slot on the same host."""
    queries = [f"rentable=true dph_total<={args.max_dph}", "rentable=true"]
    if args.machine:
        queries.insert(0, f"machine_id={args.machine}")
    twins = []
    for query in queries:
        found = vast.search_offers(
            query=query,
            type="on-demand",
            order="dph_total",
            storage=float(args.disk_gb),
            limit=1000,
        )
        if isinstance(found, dict):
            found = found.get("offers", [])
        for o in found or []:
            if int(o["id"]) == args.offer:
                return o
            if (
                args.machine
                and o.get("machine_id") == args.machine
                and int(o.get("num_gpus") or 1) == 1
            ):
                twins.append(o)
    if twins:
        twin = min(twins, key=lambda o: float(o["dph_total"]))
        print(
            f"offer {args.offer} not listed; using twin slot {twin['id']}", flush=True
        )
        return twin
    return None


def rent(argv):
    ap = argparse.ArgumentParser(prog="r2_pull_bench rent")
    ap.add_argument("--offer", type=int, required=True, help="offer id you approved")
    ap.add_argument("--max-dph", type=float, required=True, help="refuse above $/hr")
    ap.add_argument(
        "--machine",
        type=int,
        help="machine id of the approved offer: a host lists one offer per GPU "
        "slot, so accept another single-GPU offer on it at or under --max-dph",
    )
    ap.add_argument("--disk-gb", type=int, default=60)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--variants", nargs="+", type=parse_variant, required=True)
    args = ap.parse_args(argv)

    from dotenv import dotenv_values
    from vastai import VastAI

    from tools import start_sam as S

    root = Path(__file__).resolve().parents[1]
    env = {**dotenv_values(root / ".env"), **os.environ}
    missing = [k for k in ("VAST_API_KEY", "SAM_SSH_KEY", *R2_KEYS) if not env.get(k)]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}")
    S.IMAGE = IMAGE
    label = f"{LABEL_PREFIX}{uuid4().hex[:10]}"
    vast = VastAI(api_key=env["VAST_API_KEY"], retry=1)

    def interrupt(*_):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    instance_id = None
    ssh_dir = tempfile.TemporaryDirectory(prefix="r2bench-ssh-")
    try:
        key = S.prepare_ssh_key(
            Path(env["SAM_SSH_KEY"]).expanduser(), Path(ssh_dir.name)
        )
        offer = find_approved_offer(vast, args)
        if offer is None:
            raise SystemExit("approved offer is gone; nothing rented")
        if float(offer["dph_total"]) > args.max_dph:
            raise SystemExit(
                f"offer now ${offer['dph_total']:.3f}/hr > cap; nothing rented"
            )
        print(
            f"offer {offer['id']} {offer.get('gpu_name')} {offer.get('geolocation')} "
            f"${float(offer['dph_total']):.3f}/hr",
            flush=True,
        )
        instance_id = S.create_from_offers(
            vast, [offer], disk_gb=args.disk_gb, label=label
        )
        pub = subprocess.run(
            ["ssh-keygen", "-y", "-f", str(key)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(10):  # team key: attach at minute 0 (see vast-gpu-rent)
            try:
                vast.attach_ssh(instance_id=instance_id, ssh_key=pub)
                break
            except Exception as e:
                print(f"attach retry ({type(e).__name__})", flush=True)
                time.sleep(6)
        opts = [
            "-i",
            str(key),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ServerAliveInterval=30",
        ]
        host, port = S.wait_for_ssh(vast, instance_id, opts)
        remote = "/root/r2bench"
        S.run(
            [
                "ssh",
                *opts,
                "-p",
                str(port),
                f"root@{host}",
                f"mkdir -p {remote}/tools/unified_benchmark && "
                f"touch {remote}/tools/__init__.py && "
                "python3 -m pip install -q boto3 PyYAML >/dev/null 2>&1",
            ],
            timeout=600,
        )
        S.run(
            [
                "scp",
                *opts,
                "-P",
                str(port),
                *(str(root / "tools" / name) for name in BOX_TOOLS),
                f"root@{host}:{remote}/tools/",
            ],
            timeout=120,
        )
        S.run(
            [
                "scp",
                *opts,
                "-P",
                str(port),
                *(str(root / "tools" / name) for name in BOX_TOOLS_UB),
                f"root@{host}:{remote}/tools/unified_benchmark/",
            ],
            timeout=120,
        )
        S.run(
            [
                "scp",
                *opts,
                "-P",
                str(port),
                str(Path(__file__).resolve()),
                f"root@{host}:{remote}/",
            ],
            timeout=120,
        )
        variants = " ".join(
            f"{k}:{a}" if b is None else f"{k}:{a}x{b}" for k, a, b in args.variants
        )
        creds = json.dumps({k: env[k] for k in R2_KEYS}) + "\n"
        S.run(
            [
                "ssh",
                *opts,
                "-p",
                str(port),
                f"root@{host}",
                f"cd {remote} && python3 -u r2_pull_bench.py box --repo {args.repo} "
                f"--revision {args.revision} {variants}",
            ],
            input=creds,
            timeout=3600,
        )
    finally:
        if instance_id is not None:
            S.release_instance(vast, instance_id)
        else:  # a lost create response may still have made an instance
            for inst in vast.show_instances() or []:
                if inst.get("label") == label:
                    S.release_instance(vast, int(inst["id"]))
        left = [
            i.get("id")
            for i in vast.show_instances() or []
            if str(i.get("label", "")).startswith(LABEL_PREFIX)
        ]
        print(f"remaining {LABEL_PREFIX}* instances: {left}", flush=True)
        ssh_dir.cleanup()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in {"box", "rent"}:
        raise SystemExit("usage: r2_pull_bench.py {rent,box} ...")
    (box if argv[0] == "box" else rent)(argv[1:])


if __name__ == "__main__":
    main()
