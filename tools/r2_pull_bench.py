"""A/B benchmark of R2 weight-download strategies on one rented Vast box.

Answers "is the remaining ceiling our Python downloader or the network?" by
pulling the same pinned model several ways on the same machine:

    threads:FxS   model_source.pull_from_r2, F files in threads x S streams
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
IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"  # has python3 + pip
# Pinned s5cmd release; hash from the release's s5cmd_checksums.txt.
S5CMD_URL = (
    "https://github.com/peak/s5cmd/releases/download/v2.3.0/"
    "s5cmd_2.3.0_Linux-64bit.tar.gz"
)
S5CMD_SHA256 = "de0fdbfa3aceae55e069ba81a0fc17b2026567637603734a387b2fca06c299b4"


def parse_variant(text):
    kind, _, shape = text.partition(":")
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
    ap.add_argument("variants", nargs="+", type=parse_variant)
    args = ap.parse_args(argv)
    os.environ.update(json.loads(sys.stdin.readline()))  # R2_* only, via stdin
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import model_source as ms

    print(f"cpus={os.cpu_count()} disk={shutil.disk_usage(here).free / 1e9:.0f}GB")
    out = here / "bench-out"
    for kind, a, b in args.variants:
        shutil.rmtree(out, ignore_errors=True)
        for name in ("R2_PULL_FILE_WORKERS", "R2_PULL_STREAMS", "R2_PULL_PROCESSES"):
            os.environ.pop(name, None)
        note = ""
        t = time.monotonic()
        try:
            if kind == "s5cmd":
                manifest, download = _pull_s5cmd(
                    ms, args.repo, args.revision, out, a, b, here
                )
                note = f" (download {download:.0f}s + verify)"
            else:
                key = "R2_PULL_PROCESSES" if kind == "procs" else "R2_PULL_FILE_WORKERS"
                os.environ[key] = str(a)
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


# ------------------------------------------------------------------ laptop side


def rent(argv):
    ap = argparse.ArgumentParser(prog="r2_pull_bench rent")
    ap.add_argument("--offer", type=int, required=True, help="offer id you approved")
    ap.add_argument("--max-dph", type=float, required=True, help="refuse above $/hr")
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
        offer = None
        for query in (f"id={args.offer}", "rentable=true"):
            found = vast.search_offers(
                query=query,
                type="on-demand",
                order="dph_total",
                storage=float(args.disk_gb),
                limit=1000,
            )
            if isinstance(found, dict):
                found = found.get("offers", [])
            offer = next((o for o in found or [] if int(o["id"]) == args.offer), None)
            if offer is not None:
                break
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
                f"mkdir -p {remote} && python3 -m pip install -q boto3 >/dev/null 2>&1",
            ],
            timeout=600,
        )
        S.run(
            [
                "scp",
                *opts,
                "-P",
                str(port),
                str(Path(__file__).resolve()),
                str(root / "tools" / "model_source.py"),
                f"root@{host}:{remote}/",
            ],
            timeout=120,
        )
        variants = " ".join(f"{k}:{a}x{b}" for k, a, b in args.variants)
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
