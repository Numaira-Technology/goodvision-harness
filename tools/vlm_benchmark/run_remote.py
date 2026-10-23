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


def bundle(
    root, videos, destination, *, include_video=True, scope="external_vlm_api_only"
):
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
        if p.suffix in {".py", ".sh", ".json", ".txt", ".html", ".js"}
    ]
    for directory in (
        "videojev_probe", "videojev_benchmark", "videojev_native",
        "videojev_walltime", "videojev_vllm",
    ):
        files += [
            p for p in (root / "tools" / directory).iterdir()
            if p.suffix in {".py", ".txt"}
        ]
    manifest = {"files": {}, "videos": [], "scope": scope}
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
        if sys.platform == "darwin":
            self.options += ["-o", "UseKeychain=yes"]
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

    def call(self, code, *args, data=None, capture=True, timeout=60):
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
            timeout=timeout,
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

# Embed stdlib-only helpers so recovery also works with older uploaded bundles
# and before the instance has installed any benchmark dependencies.
REMOTE_LIFECYCLE = Path(__file__).with_name("lifecycle.py").read_text(encoding="utf-8")

LAUNCH = (
    REMOTE_LIFECYCLE
    + """
import os,sys,json,pathlib,tarfile,subprocess,shutil
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
plan=json.loads((p/'plan.json').read_text()) if (p/'plan.json').exists() else {}
managed=plan.get('suite')=='modes' and shutil.which('supervisorctl')
control={'pid':None,'root':str(app),'launch_state':'preparing'}
if managed: control['service']='gv-vlm-'+p.name
save_control_json(p/'control.json',control)
save_control_json(p/'status.json',{'state':'starting','job':'launch'})
try:
 (p/'runtime-key.json').write_text(sys.stdin.read())
 (p/'runtime-key.json').chmod(0o600)
 if managed:
  sys.path.insert(0,str(app))
  from tools.vlm_benchmark.cloud_modes import launch_managed
  control.update(launch_managed(p))
 else:
  with (p/'setup-and-run.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,'-m','tools.vlm_benchmark.bootstrap','--root',str(app)],
    cwd=app,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
  control['pid']=proc.pid
 control['launch_state']='started'
 save_control_json(p/'control.json',control)
 if (p/'launch-failure.json').exists(): raise RuntimeError('cloud launch was cancelled')
 try:
  fields=pathlib.Path('/proc/%s/stat'%control['pid']).read_text().rsplit(')',1)[1].split()
  control['process_start_ticks']=fields[19]
 except FileNotFoundError: pass
 save_control_json(p/'control.json',control)
 save_control_json(p/'launched.json',{'pid':control['pid']})
except BaseException as exc:
 record_launch_failure(p,control,'launch_failed:'+type(exc).__name__)
 raise SystemExit(2) from None
"""
)

STATUS = (
    REMOTE_LIFECYCLE
    + """
import json,sys,pathlib,os
p=pathlib.Path(sys.argv[1]); status=p/'status.json'
failure=p/'launch-failure.json'
if failure.exists():
 print('VLM_STATUS='+failure.read_text())
 raise SystemExit(0)
if not (p/'control.json').exists():
 data=record_launch_failure(p,recovery_control(p),'launch_control_missing')
 print('VLM_STATUS='+json.dumps(data))
 raise SystemExit(0)
data=(json.loads(status.read_text()) if status.exists()
 else {'state':'starting','job':'environment'})
if data['state'] not in ('completed','failed','cancelled'):
 control=json.loads((p/'control.json').read_text()); pid=control['pid']
 if not pid:
  data=record_launch_failure(p,control,'launch_not_finalized')
  print('VLM_STATUS='+json.dumps(data))
  raise SystemExit(0)
 try:
  fields=pathlib.Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()
  raw=pathlib.Path('/proc/%s/cmdline'%pid).read_bytes()
  if control.get('process_start_ticks'):
   alive=fields[0]!='Z' and fields[19]==control['process_start_ticks']
  else:
   # execve can temporarily expose an empty command line for a live process.
   alive=fields[0]!='Z' and (not raw or (
    b'tools.vlm_benchmark.' in raw and str(p/'app').encode() in raw))
 except OSError: alive=False
 if not alive:
  data.update(state='failed',reason='process_exited_without_completion')
  temp=status.with_suffix('.tmp'); temp.write_text(json.dumps(data))
  os.replace(temp,status)
  (p/'runtime-key.json').unlink(missing_ok=True)
  sys.path.insert(0,str(p/'app'))
  output=p/'results'; output.mkdir(exist_ok=True)
  plan=json.loads((p/'plan.json').read_text())
  if plan.get('suite')=='modes':
   from tools.vlm_benchmark.cloud_modes import failure_report
   failure_report(output,plan,'failed',data['reason'])
  else:
   from tools.vlm_benchmark.report import render
   summary=output/'summary.json'
   doc=json.loads(summary.read_text()) if summary.exists() else {'plan':plan,'jobs':[]}
   doc.update(state='failed',reason=data['reason'],engineering_capacity_gate_passed=False,
    conclusion='Process exited; unfinished stages are not acceptance evidence.')
   render(output,doc)
print('VLM_STATUS='+json.dumps(data))
"""
)

STOP = (
    REMOTE_LIFECYCLE
    + """
p=Path(sys.argv[1]); control=recovery_control(p)
(p/'runtime-key.json').unlink(missing_ok=True)
failure=p/'launch-failure.json'
if failure.exists() or not control.get('pid'):
 reason=(json.loads(failure.read_text())['reason'] if failure.exists()
  else 'launch_control_missing')
 data=record_launch_failure(p,control,reason)
 if data['cleanup_required']: raise RuntimeError('could not confirm stop; retry --stop')
else:
 stop_control(p,control)
"""
)

COLLECT = """
import pathlib,sys,tarfile
p=pathlib.Path(sys.argv[1])
with tarfile.open(p/'results.tar.gz','w:gz') as tar:
 for source in sorted((p/'results').rglob('*')):
  if (source.is_file() and not source.is_symlink()
   and source.suffix in ('.json','.jsonl','.html','.mp4','.webm','.png','.csv')):
   tar.add(source,arcname=source.relative_to(p).as_posix())
 for name in ('status.json','launch-failure.json','plan.json',
  'setup-and-run.log','host-guide.txt'):
  source=p/name
  if source.is_file(): tar.add(source,arcname=name)
"""


def watch(ssh, remote):
    previous = None
    monitor_url = None
    while True:
        response = ssh.call(STATUS, remote)
        lines = [
            line for line in response.splitlines() if line.startswith("VLM_STATUS=")
        ]
        if not lines:
            raise RuntimeError("missing remote status")
        status = json.loads(lines[-1].split("=", 1)[1])
        if status.get("monitor_url") and status["monitor_url"] != monitor_url:
            monitor_url = status["monitor_url"]
            print(
                f"Live monitor (SSH port forwarding required): {monitor_url}",
                flush=True,
            )
        update = (status["state"], status.get("job"), status.get("reason"))
        if update != previous:
            print(f"{update[0]}: {update[1]} {update[2] or ''}", flush=True)
            previous = update
        if status["state"] in {"completed", "failed", "cancelled"}:
            if status.get("cleanup_required"):
                print(
                    "Cloud launch failed; stop is not confirmed. Retry --stop.",
                    file=sys.stderr,
                )
                raise RuntimeError("cloud cleanup required")
            return status
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["capacity", "modes"], default="capacity")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--identity", type=Path, default=Path.home() / ".ssh/id_ed25519_vast"
    )
    parser.add_argument("--video", type=Path, action="append")
    parser.add_argument("--reuse-video", metavar="RUN_ID")
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 4, 8, 12])
    parser.add_argument("--provider", choices=["openrouter", "dashscope"])
    parser.add_argument("--model", default="")
    parser.add_argument("--soak-seconds", type=int)
    parser.add_argument("--global-limit", type=int)
    parser.add_argument("--per-stream-limit", type=int)
    parser.add_argument("--request-rate", type=float)
    parser.add_argument(
        "--real", action="store_true", help="modes suite: call paid API"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["events", "grounding", "compact", "parallel"],
        default=["events", "compact", "parallel"],
    )
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-seconds", type=float, default=3)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor-port", type=int, default=8787)
    parser.add_argument(
        "--monitor-layout", choices=["evidence", "single-video"], default="evidence"
    )
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
    parser.add_argument("--arrival-mode", choices=["aligned", "staggered"])
    parser.add_argument(
        "--event-json-mode", choices=["prompt", "schema"], default="prompt"
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", metavar="RUN_ID")
    action.add_argument("--stop", metavar="RUN_ID")
    args = parser.parse_args()
    mode_suite = args.suite == "modes"
    for name, value in {
        "provider": "openrouter",
        "soak_seconds": 0 if mode_suite else 900,
        "global_limit": 48 if mode_suite else 16,
        "per_stream_limit": 4 if mode_suite else 2,
        "request_rate": 40 if mode_suite else 20,
        "arrival_mode": "staggered" if mode_suite else "aligned",
    }.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    try:
        if mode_suite:
            from tools.vlm_benchmark.cloud_modes import make_plan as make_modes_plan

            if args.provider != "openrouter" or any(
                (
                    args.compare,
                    args.compare_concurrency,
                    args.compare_arrival,
                    args.compare_routing,
                    args.provider_only,
                    args.provider_sort,
                )
            ):
                parser.error("modes suite uses OpenRouter and its fixed routing policy")
            args.model = args.model or "qwen/qwen3-vl-30b-a3b-instruct"
            plan = make_modes_plan(args)
        else:
            if args.monitor_layout != "evidence":
                parser.error("monitor layout requires --suite modes")
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
    if mode_suite:
        print(
            "Three-mode suite: 3 s is a latency reference, not a business SLA.",
            flush=True,
        )
    else:
        print(
            "Proposed gates: each stream >=95% complete results within 3 s; "
            "errors <=1%; "
            "uncertain judgments <=10%. Not a semantic-accuracy acceptance.",
            flush=True,
        )
    if not args.resume and not args.stop:
        budget = plan["request_budget"] if mode_suite else request_budget(plan)
        print("Request upper budget:", json.dumps(budget), flush=True)
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
            # Read the instance guide before deploying, not just into an archive.
            guide = ssh.call(
                "from pathlib import Path; p=Path('/etc/vast-agents-guide.md'); "
                "print(p.read_text() if p.exists() else 'No instance guide found.')"
            )
            print(guide, flush=True)
            if mode_suite:
                from dotenv import dotenv_values

                key = (
                    (
                        os.environ.get("OPENROUTER_API_KEY")
                        or dotenv_values(ROOT / ".env").get("OPENROUTER_API_KEY")
                    )
                    if args.real
                    else ""
                )
            else:
                key = os.environ.get("INFERENCE_API_KEY") or os.environ.get(
                    {
                        "openrouter": "OPENROUTER_API_KEY",
                        "dashscope": "DASHSCOPE_API_KEY",
                    }[args.provider]
                )
            if not plan["simulate"]:
                key = key or getpass.getpass(f"{args.provider} API key (hidden): ")
            if not plan["simulate"] and not key.strip():
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
        # Recorded suites contain videos; packaging can exceed an RPC's 60 s.
        ssh.call(COLLECT, remote, timeout=1800)
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
