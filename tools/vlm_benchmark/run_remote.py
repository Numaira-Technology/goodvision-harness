"""Package, install, run and retrieve one cloud VLM test; never rents a GPU."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.vlm_benchmark.plan import make_plan, request_budget  # noqa: E402

REMOTE = "/workspace/goodvision-vlm-tests"
RUN_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}\Z")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def bundle(root, videos, destination, *, include_video=True):
    root = Path(root).resolve()
    files = list((root / "src/goodvision_harness").rglob("*.py"))
    files += [
        root / "src/goodvision_harness/config.yaml",
        root / "pyproject.toml",
        root / "tools/vlm_multistream_benchmark.py",
    ]
    files += [
        p
        for p in (root / "tools/vlm_benchmark").iterdir()
        if p.suffix in {".py", ".sh", ".json", ".txt"}
    ]
    manifest = {"files": {}, "videos": [], "scope": "external_vlm_api_only"}
    with tempfile.TemporaryDirectory(prefix="vlm-package-") as temp:
        stage = Path(temp)
        for source in sorted(set(files)):
            if source.is_symlink() or not source.resolve().is_relative_to(root):
                raise ValueError("source symlinks are not allowed")
            relative = source.relative_to(root)
            if any(
                p.startswith(".env") or p in {".git", "__pycache__"}
                for p in relative.parts
            ):
                raise ValueError("secret/cache path in source selection")
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            manifest["files"][relative.as_posix()] = digest(target)
        for index, source in enumerate(videos, 1):
            source = Path(source)
            if not source.is_file() or source.is_symlink():
                raise ValueError("video must be an existing regular file")
            name = f"media/video-{index:02d}{source.suffix.lower()}"
            target = stage / name
            target.parent.mkdir(exist_ok=True)
            if include_video:
                shutil.copyfile(source, target)
            manifest["videos"].append(name)
            manifest["files"][name] = digest(source)
        (stage / "bundle-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        with tarfile.open(destination, "w:gz") as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(stage).as_posix())
    return manifest


def safe_extract(archive_path, destination):
    """No traversal, links, devices or surprising file types in downloaded results."""
    destination = Path(destination).resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            path = destination / member.name
            if (
                not member.isfile()
                or not path.resolve().is_relative_to(destination)
                or Path(member.name).is_absolute()
                or path.is_symlink()
            ):
                raise ValueError("unsafe result archive")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)


class SSH:
    @staticmethod
    def validate(args):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", args.host or ""):
            raise ValueError("--host must be an IP address or hostname")
        if not args.port or not 1 <= args.port <= 65535:
            raise ValueError("--port must be in 1..65535")
        if not args.identity.expanduser().is_file():
            raise ValueError("SSH identity file not found")

    def __init__(self, args):
        self.validate(args)
        self.temp = tempfile.TemporaryDirectory(prefix="gvvlm-", dir="/tmp")
        self.socket = str(Path(self.temp.name) / "s")
        self.host = f"root@{args.host}"
        self.options = [
            "-F",
            "/dev/null",
            "-i",
            str(args.identity.expanduser()),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            f"ControlPath={self.socket}",
        ]
        self.port = str(args.port)

    def connect(self):
        subprocess.run(
            [
                "ssh",
                *self.options,
                "-p",
                self.port,
                "-M",
                "-N",
                "-f",
                "-o",
                "BatchMode=no",
                self.host,
            ],
            check=True,
            timeout=180,
        )

    def call(self, code, *args, data=None, capture=True):
        command = shlex.join(["python3", "-c", code, *map(str, args)])
        result = subprocess.run(
            [
                "ssh",
                *self.options,
                "-p",
                self.port,
                "-o",
                "BatchMode=yes",
                self.host,
                command,
            ],
            input=data,
            stdout=subprocess.PIPE if capture else None,
            text=True,
            check=True,
            timeout=60,
        )
        return result.stdout

    def copy(self, source, destination):
        subprocess.run(
            ["scp", *self.options, "-P", self.port, str(source), str(destination)],
            check=True,
            timeout=1800,
        )

    def close(self):
        subprocess.run(
            ["ssh", *self.options, "-p", self.port, "-O", "exit", self.host],
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.temp.cleanup()


CREATE = """
import os,sys,pathlib
os.umask(0o077)
p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=False)
guide=pathlib.Path('/etc/vast-agents-guide.md')
if guide.exists(): (p/'host-guide.txt').write_text(guide.read_text())
"""

LAUNCH = """
import os,sys,json,pathlib,tarfile,subprocess
os.umask(0o077)
p=pathlib.Path(sys.argv[1]); app=p/'app'; app.mkdir()
with tarfile.open(p/'bundle.tar.gz') as tar:
 for m in tar.getmembers():
  target=app/m.name
  if not m.isfile() or not target.resolve().is_relative_to(app):
   raise ValueError('unsafe bundle')
  target.parent.mkdir(parents=True,exist_ok=True)
  with tar.extractfile(m) as source, target.open('wb') as dest:
   import shutil; shutil.copyfileobj(source,dest)
if len(sys.argv)>2 and sys.argv[2]:
 import shutil
 previous=pathlib.Path(sys.argv[2])/'app'
 manifest=json.loads((app/'bundle-manifest.json').read_text())
 for name in manifest['videos']:
  source=previous/name; target=app/name
  if (not source.is_file() or source.is_symlink()
   or not source.resolve().is_relative_to(previous)):
   raise ValueError('reusable video missing')
  target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(source,target)
(p/'runtime-key.json').write_text(sys.stdin.read()); (p/'runtime-key.json').chmod(0o600)
with (p/'setup-and-run.log').open('w') as log:
 proc=subprocess.Popen([sys.executable,'-m','tools.vlm_benchmark.bootstrap','--root',str(app)],
  cwd=app,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(p/'control.json').write_text(json.dumps({'pid':proc.pid,'root':str(app)}))
(p/'launched.json').write_text(json.dumps({'pid':proc.pid}))
"""

STATUS = """
import json,sys,pathlib,os
p=pathlib.Path(sys.argv[1]); status=p/'status.json'
data=(json.loads(status.read_text()) if status.exists()
 else {'state':'starting','job':'environment'})
if data['state'] not in ('completed','failed','cancelled'):
 pid=json.loads((p/'control.json').read_text())['pid']
 try:
  raw=pathlib.Path('/proc/%s/cmdline'%pid).read_bytes()
  alive=b'tools.vlm_benchmark.' in raw and str(p/'app').encode() in raw
 except OSError: alive=False
 if not alive:
  data.update(state='failed',reason='process_exited_without_completion')
  temp=status.with_suffix('.tmp'); temp.write_text(json.dumps(data))
  os.replace(temp,status)
  (p/'runtime-key.json').unlink(missing_ok=True)
  sys.path.insert(0,str(p/'app'))
  from tools.vlm_benchmark.report import render
  output=p/'results'; output.mkdir(exist_ok=True)
  summary=output/'summary.json'
  doc=json.loads(summary.read_text()) if summary.exists() else {
   'plan':json.loads((p/'plan.json').read_text()),'jobs':[]}
  doc.update(state='failed',reason=data['reason'],engineering_capacity_gate_passed=False,
   conclusion='Process exited; unfinished stages are not acceptance evidence.')
  render(output,doc)
print('VLM_STATUS='+json.dumps(data))
"""

STOP = """
import os,signal,json,sys,pathlib
p=pathlib.Path(sys.argv[1]); pid=json.loads((p/'control.json').read_text())['pid']
try:
 raw=pathlib.Path('/proc/%s/cmdline'%pid).read_bytes()
 if (b'tools.vlm_benchmark.' not in raw
  or str(p/'app').encode() not in raw or os.getpgid(pid)!=pid):
  raise RuntimeError('refusing to stop an unrelated process')
 os.killpg(pid,signal.SIGTERM)
except (FileNotFoundError,ProcessLookupError): pass
(p/'runtime-key.json').unlink(missing_ok=True)
"""

COLLECT = """
import pathlib,sys,tarfile
p=pathlib.Path(sys.argv[1])
with tarfile.open(p/'results.tar.gz','w:gz') as tar:
 for source in sorted((p/'results').rglob('*')):
  if (source.is_file() and not source.is_symlink()
   and source.suffix in ('.json','.jsonl','.html')):
   tar.add(source,arcname=source.relative_to(p).as_posix())
 for name in ('status.json','plan.json','setup-and-run.log','host-guide.txt'):
  source=p/name
  if source.is_file(): tar.add(source,arcname=name)
"""


def watch(ssh, remote):
    previous = None
    while True:
        response = ssh.call(STATUS, remote)
        lines = [
            line for line in response.splitlines() if line.startswith("VLM_STATUS=")
        ]
        if not lines:
            raise RuntimeError("missing remote status")
        status = json.loads(lines[-1].split("=", 1)[1])
        update = (status["state"], status.get("job"), status.get("reason"))
        if update != previous:
            print(f"{update[0]}: {update[1]} {update[2] or ''}", flush=True)
            previous = update
        if status["state"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--identity", type=Path, default=Path.home() / ".ssh/id_ed25519_vast"
    )
    parser.add_argument("--video", type=Path, action="append")
    parser.add_argument("--reuse-video", metavar="RUN_ID")
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument(
        "--provider", choices=["aiclub", "openrouter", "dashscope"], default="aiclub"
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--soak-seconds", type=int, default=900)
    parser.add_argument("--global-limit", type=int, default=16)
    parser.add_argument("--per-stream-limit", type=int, default=2)
    parser.add_argument("--request-rate", type=float, default=20)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-concurrency", action="store_true")
    parser.add_argument("--compare-arrival", action="store_true")
    parser.add_argument("--compare-routing", action="store_true")
    parser.add_argument(
        "--provider-only", choices=["deepinfra", "parasail", "venice"], default=""
    )
    parser.add_argument(
        "--provider-sort", choices=["", "throughput", "latency"], default=""
    )
    parser.add_argument(
        "--arrival-mode", choices=["aligned", "staggered"], default="aligned"
    )
    parser.add_argument(
        "--event-json-mode", choices=["prompt", "schema"], default="prompt"
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", metavar="RUN_ID")
    action.add_argument("--stop", metavar="RUN_ID")
    args = parser.parse_args()
    try:
        plan = make_plan(
            provider=args.provider,
            model=args.model,
            soak_seconds=args.soak_seconds,
            global_limit=args.global_limit,
            per_stream_limit=args.per_stream_limit,
            request_rate=args.request_rate,
            compare=args.compare,
            streams=args.streams,
            compare_concurrency=args.compare_concurrency,
            compare_arrival=args.compare_arrival,
            compare_routing=args.compare_routing,
            provider_sort=args.provider_sort,
            provider_only=args.provider_only,
            arrival_mode=args.arrival_mode,
            event_json_mode=args.event_json_mode,
        )
        if args.reuse_video and not RUN_ID.fullmatch(args.reuse_video):
            parser.error("invalid reusable video run ID")
        if not args.prepare_only and not args.plan_only:
            SSH.validate(args)
        if (args.resume or args.stop) and (args.prepare_only or args.plan_only):
            parser.error("resume/stop cannot be used with local preparation flags")
    except ValueError as exc:
        parser.error(str(exc))
    print("External VLM API test: the rented GPU does not run inference.", flush=True)
    print(
        "Proposed gates: each stream >=95% complete results within 3 s; errors <=1%; "
        "uncertain judgments <=10%. Not a semantic-accuracy acceptance.",
        flush=True,
    )
    if not args.resume and not args.stop:
        print("Request upper budget:", json.dumps(request_budget(plan)), flush=True)
        if plan.get("compare_concurrency"):
            print(
                "One stream; in-flight request limits 2 -> 4 -> 4 -> 2; "
                "30 s each. No twelve-stream or soak stage.",
                flush=True,
            )
        if plan.get("compare_arrival"):
            print(
                "Arrival ABBA: aligned -> staggered -> staggered -> aligned; "
                "same frames, source duration, sample rate and deadlines.",
                flush=True,
            )
        if plan.get("compare_routing"):
            print(
                "Routing ABBA: throughput -> latency -> latency -> throughput; "
                "same base model, inputs, schema, rate and deadlines.",
                flush=True,
            )
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    run_id = (
        args.resume
        or args.stop
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    )
    if not RUN_ID.fullmatch(run_id):
        parser.error("invalid run ID")
    local = ROOT / "outputs/vlm-cloud" / run_id
    remote = f"{REMOTE}/{run_id}"
    try:
        local.mkdir(parents=True, exist_ok=bool(args.resume or args.stop))
        if not args.resume and not args.stop:
            videos = args.video or [Path.home() / "Downloads/output 3.mp4"]
            (local / "plan.json").write_text(
                json.dumps(plan, indent=2) + "\n", encoding="utf-8"
            )
            bundle(
                ROOT,
                videos,
                local / "bundle.tar.gz",
                include_video=not args.reuse_video,
            )
            print(f"Prepared: {local}", flush=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.prepare_only:
        return 0
    ssh = SSH(args)
    started = bool(args.resume or args.stop)
    try:
        ssh.connect()
        if not started:
            key = os.environ.get("INFERENCE_API_KEY") or os.environ.get(
                {
                    "aiclub": "AICLUB_API_KEY",
                    "openrouter": "OPENROUTER_API_KEY",
                    "dashscope": "DASHSCOPE_API_KEY",
                }[args.provider]
            )
            key = key or getpass.getpass(f"{args.provider} API key (hidden): ")
            if not key.strip():
                raise ValueError("API key is empty")
            ssh.call(CREATE, remote)
            ssh.copy(local / "bundle.tar.gz", f"{ssh.host}:{remote}/bundle.tar.gz")
            ssh.copy(local / "plan.json", f"{ssh.host}:{remote}/plan.json")
            started = True
            print(f"Launching Run ID: {run_id}", flush=True)
            previous = f"{REMOTE}/{args.reuse_video}" if args.reuse_video else ""
            ssh.call(LAUNCH, remote, previous, data=json.dumps({"key": key.strip()}))
            key = None
        print(
            f"Run ID: {run_id}\nCloud job is bounded and survives connection loss. "
            "Use the same SSH options with --resume or --stop and this Run ID.",
            flush=True,
        )
        if args.stop:
            ssh.call(STOP, remote)
        status = watch(ssh, remote)
        ssh.call(COLLECT, remote)
        ssh.copy(f"{ssh.host}:{remote}/results.tar.gz", local / "results.tar.gz")
        safe_extract(local / "results.tar.gz", local)
        print(f"Report: {local / 'results/report.html'}", flush=True)
        return 0 if status["state"] == "completed" else 2
    except KeyboardInterrupt:
        if started:
            try:
                ssh.call(STOP, remote)
                print(f"Stop requested. Recover partial results with --resume {run_id}")
            except Exception:
                print(
                    "Could not confirm stop; the cloud job may still be running. "
                    f"--stop {run_id}"
                )
        return 130
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as exc:
        print(
            f"Runner stopped: {type(exc).__name__}. Run ID: {run_id}", file=sys.stderr
        )
        if started:
            print(
                f"Cloud work may continue; use --resume {run_id} or --stop {run_id}.",
                file=sys.stderr,
            )
        return 2
    finally:
        try:
            ssh.close()
        except (subprocess.SubprocessError, OSError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
